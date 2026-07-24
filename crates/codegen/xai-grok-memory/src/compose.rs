//! Compositional retrieval and injection (MemEIC Knowledge Connector analog).
//!
//! Pipeline:
//! 1. **Decompose** the user query into the memory kinds it needs
//! 2. **Retrieve** best-of-kind (parallel logical searches, merged)
//! 3. **Fuse** into a sectioned injection template that forces cross-kind
//!    attention and prefers retrieved knowledge over model priors
//!
//! This is the agent-native stand-in for MemEIC's dual external stores +
//! attention connector: we cannot LoRA the hosted model, so composition
//! happens in retrieval partitioning and prompt structure.

use std::collections::{BTreeMap, HashSet};

use super::embedding::EmbeddingProvider;
use super::index::MemoryIndex;
use super::kind::MemoryKind;
use super::search::{SearchFilter, SearchResult};
use xai_grok_config_types::MemorySearchConfig;

/// Maximum results pulled per kind during compositional search.
const PER_KIND_LIMIT: usize = 2;

/// Result of compositional retrieval, grouped by kind.
#[derive(Debug, Clone, Default)]
pub struct CompositionalResults {
    /// Kinds the query was decomposed into (may be empty → untyped fallback).
    pub requested_kinds: Vec<MemoryKind>,
    /// Results grouped by kind, insertion order preserved within each group.
    pub by_kind: BTreeMap<MemoryKind, Vec<SearchResult>>,
    /// Flat list in section order (for callers that want a single ranking).
    pub flat: Vec<SearchResult>,
}

impl CompositionalResults {
    pub fn is_empty(&self) -> bool {
        self.flat.is_empty()
    }

    pub fn len(&self) -> usize {
        self.flat.len()
    }
}

/// Rule-based query decomposition into memory kinds.
///
/// Returns an empty vec when the query has no type signals — callers then
/// fall back to untyped hybrid search (legacy behaviour).
pub fn decompose_query(query: &str) -> Vec<MemoryKind> {
    let q = query.to_ascii_lowercase();
    let mut kinds = Vec::new();

    let has_decision = contains_any(
        &q,
        &[
            "decision",
            "decided",
            "why did we",
            "why do we",
            "chose",
            "chosen",
            "architecture",
            "adr",
            "rejected",
            "instead of",
        ],
    );
    let has_procedure = contains_any(
        &q,
        &[
            "how do we",
            "how to",
            "how did we",
            "runbook",
            "procedure",
            "steps to",
            "debug",
            "rebuild",
            "workflow",
        ],
    );
    let has_preference = contains_any(
        &q,
        &[
            "prefer",
            "preference",
            "always",
            "never ",
            "habit",
            "style",
        ],
    );
    let has_fact = contains_any(
        &q,
        &[
            "what is",
            "what's",
            "fact",
            "version",
            "which crate",
            "where is",
            "api",
        ],
    );
    let has_entity = contains_any(&q, &["who is", "who owns", "service ", "ticket"]);
    let has_episode = contains_any(
        &q,
        &[
            "last session",
            "yesterday",
            "earlier today",
            "we discussed",
            "last time",
        ],
    );

    // Compositional cue: multi-hop / combine language forces multi-kind.
    let compositional = contains_any(
        &q,
        &[
            " and ",
            " both ",
            " as well as ",
            " plus ",
            "why and how",
            "decision and",
            "and what",
            "and how",
        ],
    );

    if has_decision {
        kinds.push(MemoryKind::Decision);
    }
    if has_procedure {
        kinds.push(MemoryKind::Procedure);
    }
    if has_preference {
        kinds.push(MemoryKind::Preference);
    }
    if has_fact {
        kinds.push(MemoryKind::Fact);
    }
    if has_entity {
        kinds.push(MemoryKind::Entity);
    }
    if has_episode {
        kinds.push(MemoryKind::Episode);
    }

    // If the query looks compositional but only one kind matched, broaden
    // with the complementary pair decision+procedure (most common coding case).
    if compositional && kinds.len() == 1 {
        if kinds[0] == MemoryKind::Decision {
            kinds.push(MemoryKind::Procedure);
        } else if kinds[0] == MemoryKind::Procedure {
            kinds.push(MemoryKind::Decision);
        } else if kinds[0] == MemoryKind::Fact {
            kinds.push(MemoryKind::Decision);
        }
    }

    // Explicit multi-hop coding pattern: "why … and how …"
    if compositional && kinds.is_empty() {
        kinds.push(MemoryKind::Decision);
        kinds.push(MemoryKind::Procedure);
    }

    kinds.sort();
    kinds.dedup();
    kinds
}

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Embed a query string for hybrid search (async; no index borrow).
///
/// Separated from index access so callers can keep `MemoryIndex` futures
/// `Send` (rusqlite connections are `Send + !Sync`).
pub async fn embed_query(
    embedding_provider: Option<&dyn EmbeddingProvider>,
    query: &str,
    vec_available: bool,
) -> Option<Vec<f32>> {
    if !vec_available {
        return None;
    }
    let provider = embedding_provider?;
    match provider.embed_batch(&[query]).await {
        Ok(mut embeddings) if !embeddings.is_empty() => Some(embeddings.swap_remove(0)),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!(error = %e, "embedding query failed, falling back to FTS-only");
            None
        }
    }
}

/// Run compositional (or untyped fallback) hybrid search — **sync** after
/// embeddings are prepared.
///
/// When `decompose_query` yields kinds, retrieves up to [`PER_KIND_LIMIT`]
/// results per kind with superseded chunks excluded. Otherwise runs a single
/// untyped hybrid search (legacy path).
///
/// Call [`embed_query`] first if vector search is desired. Keeping this
/// function sync ensures `MemoryIndex` is never borrowed across `.await`
/// (required for `Send` futures on the tool/backend path).
pub fn compositional_search_sync(
    index: &MemoryIndex,
    query_embedding: Option<&[f32]>,
    query: &str,
    config: &MemorySearchConfig,
) -> Result<CompositionalResults, Box<dyn std::error::Error>> {
    let requested = decompose_query(query);
    let mut out = CompositionalResults {
        requested_kinds: requested.clone(),
        ..Default::default()
    };

    if requested.is_empty() {
        let filter = SearchFilter {
            kinds: None,
            exclude_superseded: true,
        };
        let results =
            hybrid_search_sync(index, query_embedding, query, config, &filter)?;
        for r in results {
            let kind = r.kind;
            out.by_kind.entry(kind).or_default().push(r.clone());
            out.flat.push(r);
        }
        return Ok(out);
    }

    let mut seen = HashSet::new();
    let mut per_kind_config = config.clone();
    per_kind_config.max_results = PER_KIND_LIMIT;
    // Lower the floor slightly so sparse kind partitions still surface.
    if per_kind_config.min_score > 0.15 {
        per_kind_config.min_score = 0.15;
    }

    for kind in &requested {
        let filter = SearchFilter {
            kinds: Some(vec![*kind]),
            exclude_superseded: true,
        };
        let results =
            hybrid_search_sync(index, query_embedding, query, &per_kind_config, &filter)?;
        for r in results {
            if seen.insert(r.chunk_id.clone()) {
                out.by_kind.entry(*kind).or_default().push(r.clone());
                out.flat.push(r);
            }
        }
    }

    // Supersession: if any result declares supersedes pointing at another
    // result's path stem or chunk id fragment, drop the older one.
    apply_supersession(&mut out);

    // Cap total flat length to config.max_results while keeping kind diversity.
    if out.flat.len() > config.max_results {
        out.flat.truncate(config.max_results);
        // Rebuild by_kind from truncated flat.
        out.by_kind.clear();
        for r in &out.flat {
            out.by_kind.entry(r.kind).or_default().push(r.clone());
        }
    }

    Ok(out)
}

/// Async convenience wrapper for tests / current-thread runtimes.
///
/// Prefer [`compositional_search_sync`] + [`embed_query`] on the live
/// `MemoryBackend` path so futures stay `Send`.
pub async fn compositional_search(
    index: &MemoryIndex,
    embedding_provider: Option<&dyn EmbeddingProvider>,
    query: &str,
    config: &MemorySearchConfig,
) -> Result<CompositionalResults, Box<dyn std::error::Error>> {
    let emb = embed_query(embedding_provider, query, index.vec_available()).await;
    compositional_search_sync(index, emb.as_deref(), query, config)
}

/// Sync hybrid search with a precomputed query embedding (Mem-E typed filter).
pub fn hybrid_search_sync(
    index: &MemoryIndex,
    query_embedding: Option<&[f32]>,
    query: &str,
    config: &MemorySearchConfig,
    filter: &SearchFilter,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let candidate_limit = config.max_results * 3;
    let kinds = filter.kinds.as_deref();

    let mut fts_results = index
        .search_fts_filtered(query, candidate_limit, None, kinds)
        .unwrap_or_default();
    let evergreen = index
        .search_fts_filtered(
            query,
            candidate_limit,
            Some(&["global", "workspace"]),
            kinds,
        )
        .unwrap_or_default();
    let existing: HashSet<String> = fts_results.iter().map(|r| r.chunk_id.clone()).collect();
    for r in evergreen {
        if !existing.contains(&r.chunk_id) {
            fts_results.push(r);
        }
    }

    super::search::hybrid_search_merge(index, fts_results, query_embedding, config, filter)
}

/// Drop chunks that another returned chunk supersedes (by id fragment or slug).
fn apply_supersession(results: &mut CompositionalResults) {
    let superseding: Vec<String> = results
        .flat
        .iter()
        .filter_map(|r| r.supersedes.clone())
        .collect();
    if superseding.is_empty() {
        return;
    }

    results.flat.retain(|r| {
        // Keep if nothing claims to supersede this chunk.
        !superseding.iter().any(|s| {
            r.chunk_id.contains(s.as_str())
                || r.path.contains(s.as_str())
                || r.snippet.to_ascii_lowercase().contains(&format!("id: {s}"))
                || snippet_declares_id(&r.snippet, s)
        })
    });

    results.by_kind.clear();
    for r in &results.flat {
        results.by_kind.entry(r.kind).or_default().push(r.clone());
    }
}

fn snippet_declares_id(snippet: &str, id: &str) -> bool {
    for line in snippet.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("id:") {
            if rest.trim() == id {
                return true;
            }
        }
    }
    false
}

/// Format compositional results as a MemEIC-style connector injection block.
///
/// Sectioned by kind with an explicit instruction to compose across sections
/// and prefer retrieved knowledge over priors.
pub fn format_compositional_injection(results: &CompositionalResults) -> Option<String> {
    if results.is_empty() {
        return None;
    }

    let mut out = String::from(
        "## Relevant Memory from Past Sessions\n\n\
         Prefer retrieved knowledge over model priors when they conflict. \
         Compose across sections when the question needs more than one kind of knowledge.\n\n",
    );

    // Stable section order matching MemoryKind ordinal.
    let order = [
        MemoryKind::Decision,
        MemoryKind::Fact,
        MemoryKind::Procedure,
        MemoryKind::Preference,
        MemoryKind::Entity,
        MemoryKind::Episode,
        MemoryKind::Unknown,
    ];

    for kind in order {
        let Some(items) = results.by_kind.get(&kind) else {
            continue;
        };
        if items.is_empty() {
            continue;
        }
        out.push_str(&format!("### {}\n\n", kind.section_title()));
        for (i, r) in items.iter().enumerate() {
            let status_note = if r.status != "active" {
                format!(", status: {}", r.status)
            } else {
                String::new()
            };
            out.push_str(&format!(
                "{}. (score: {:.2}, source: {}{}, kind: {})\n\
                 **File:** {} (lines {}-{})\n\
                 ```\n{}\n```\n\n",
                i + 1,
                r.score,
                r.source,
                status_note,
                r.kind.as_str(),
                r.path,
                r.start_line,
                r.end_line,
                r.snippet,
            ));
        }
    }

    Some(out)
}

/// Whether a query should use compositional (multi-kind) retrieval.
pub fn is_compositional_query(query: &str) -> bool {
    decompose_query(query).len() >= 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decompose_decision_only() {
        let k = decompose_query("Why did we decide to use worktrees?");
        assert_eq!(k, vec![MemoryKind::Decision]);
    }

    #[test]
    fn decompose_compositional_why_and_how() {
        let k = decompose_query(
            "Why did we decide to vendor kqueue and how do we rebuild on OpenBSD?",
        );
        assert!(k.contains(&MemoryKind::Decision), "{k:?}");
        assert!(k.contains(&MemoryKind::Procedure), "{k:?}");
        assert!(k.len() >= 2);
    }

    #[test]
    fn decompose_empty_for_generic() {
        let k = decompose_query("fix the flaky pager test");
        assert!(k.is_empty(), "generic coding query should be untyped: {k:?}");
    }

    #[test]
    fn is_compositional_detects_multi() {
        assert!(is_compositional_query(
            "Why did we choose Graphite and how do we restack?"
        ));
        assert!(!is_compositional_query("What is the session id format?"));
    }

    #[test]
    fn format_empty_is_none() {
        assert!(format_compositional_injection(&CompositionalResults::default()).is_none());
    }

    #[test]
    fn format_sections_by_kind() {
        let mut results = CompositionalResults::default();
        results.requested_kinds = vec![MemoryKind::Decision, MemoryKind::Procedure];
        results.flat = vec![
            SearchResult {
                chunk_id: "a:0".into(),
                path: "MEMORY.md".into(),
                start_line: 0,
                end_line: 4,
                score: 0.9,
                snippet: "Use worktrees for isolation.".into(),
                source: "workspace".into(),
                created_at: 1,
                kind: MemoryKind::Decision,
                supersedes: None,
                status: "active".into(),
            },
            SearchResult {
                chunk_id: "b:0".into(),
                path: "MEMORY.md".into(),
                start_line: 10,
                end_line: 20,
                score: 0.85,
                snippet: "cargo build -p xai-grok-memory".into(),
                source: "workspace".into(),
                created_at: 1,
                kind: MemoryKind::Procedure,
                supersedes: None,
                status: "active".into(),
            },
        ];
        for r in &results.flat {
            results.by_kind.entry(r.kind).or_default().push(r.clone());
        }

        let text = format_compositional_injection(&results).unwrap();
        assert!(text.contains("Retrieved decisions"));
        assert!(text.contains("Retrieved procedures"));
        assert!(text.contains("Prefer retrieved knowledge"));
        assert!(text.contains("Compose across sections"));
        // Decisions section should appear before procedures.
        let d = text.find("Retrieved decisions").unwrap();
        let p = text.find("Retrieved procedures").unwrap();
        assert!(d < p);
    }
}
