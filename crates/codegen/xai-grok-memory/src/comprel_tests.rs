//! CompRel-style fixture suite for typed / compositional memory.
//!
//! Mirrors MemEIC's Compositional Reliability idea for coding-agent memory:
//! after sequential "edits" (typed memory writes), a multi-hop query must
//! retrieve **both** a decision and a procedure (or fact) — not just the
//! single highest-scoring blended hit.

#![cfg(test)]

use crate::compose::{
    compositional_search, format_compositional_injection, is_compositional_query,
};
use crate::index::{MemoryIndex, init_sqlite_vec};
use crate::kind::MemoryKind;
use crate::search::{SearchFilter, hybrid_search_filtered};
use crate::storage::MemoryStorage;
use tempfile::TempDir;
use xai_grok_config_types::{MemoryIndexConfig, MemorySearchConfig};

fn test_index(tmp: &TempDir) -> MemoryIndex {
    init_sqlite_vec();
    let global = tmp.path().join("memory");
    let workspace = global.join("ws");
    let storage = MemoryStorage::with_paths(global, workspace);
    let db_path = tmp.path().join("index.sqlite");
    MemoryIndex::open_or_create(&db_path, storage, MemoryIndexConfig::default(), 4).unwrap()
}

/// Seed sequential typed "edits" into workspace MEMORY.md and reindex.
fn seed_typed_memory(idx: &mut MemoryIndex, path: &std::path::Path) {
    let content = r#"# Project Memory

## Decision: vendor kqueue for OpenBSD
type: decision
status: active
id: decision-vendor-kqueue

We vendor kqueue under vendor/ so OpenBSD builds do not depend on crates.io
availability for notify backends.

## Procedure: rebuild memory crate on OpenBSD
type: procedure
status: active
id: procedure-rebuild-memory

1. cd grok-build
2. cargo test -p xai-grok-memory
3. cargo build -p xai-grok-memory

## Fact: memory schema version is 2
type: fact
status: active
id: fact-schema-v2

The memory index schema_version is 2 (typed kind/supersedes/status columns).

## Decision: shared cwd for subagents (superseded)
type: decision
status: superseded
id: decision-shared-cwd

Old approach: run subagents in the parent cwd. Superseded by worktree isolation.

## Decision: use worktree isolation for execute-plan
type: decision
status: active
supersedes: decision-shared-cwd
id: decision-worktree

execute-plan subagents run in isolated git worktrees so parent state stays clean.

## Preference: always commit after green tests
type: preference
status: active

User wants agents to commit when compile + regress pass, without hourly nagging.
"#;
    std::fs::write(path, content).unwrap();
    idx.reindex_file(path, "workspace").unwrap();
}

#[test]
fn reindex_classifies_kinds() {
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let decision = idx
        .search_fts_by_kinds("vendor kqueue", 5, &[MemoryKind::Decision])
        .unwrap();
    assert!(
        !decision.is_empty(),
        "should find decision via kind-filtered FTS"
    );
    let rec = idx.get_chunk(&decision[0].chunk_id).unwrap().unwrap();
    assert_eq!(rec.kind, MemoryKind::Decision);

    let procedure = idx
        .search_fts_by_kinds("rebuild memory", 5, &[MemoryKind::Procedure])
        .unwrap();
    assert!(!procedure.is_empty());
    let rec = idx.get_chunk(&procedure[0].chunk_id).unwrap().unwrap();
    assert_eq!(rec.kind, MemoryKind::Procedure);
}

#[tokio::test]
async fn kind_filter_excludes_other_kinds() {
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let config = MemorySearchConfig {
        min_score: 0.0,
        max_results: 10,
        ..Default::default()
    };
    let filter = SearchFilter {
        kinds: Some(vec![MemoryKind::Procedure]),
        exclude_superseded: true,
    };
    let results = hybrid_search_filtered(&idx, None, "OpenBSD rebuild kqueue", &config, &filter)
        .await
        .unwrap();
    assert!(
        !results.is_empty(),
        "procedure filter should still hit rebuild procedure"
    );
    for r in &results {
        assert_eq!(r.kind, MemoryKind::Procedure, "got {:?}", r.kind);
    }
}

#[tokio::test]
async fn exclude_superseded_drops_old_decision() {
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let config = MemorySearchConfig {
        min_score: 0.0,
        max_results: 20,
        ..Default::default()
    };
    let filter = SearchFilter {
        kinds: Some(vec![MemoryKind::Decision]),
        exclude_superseded: true,
    };
    let results = hybrid_search_filtered(&idx, None, "subagents cwd worktree", &config, &filter)
        .await
        .unwrap();
    assert!(
        results
            .iter()
            .all(|r| r.status != "superseded"),
        "superseded decisions must be filtered: {:?}",
        results.iter().map(|r| (&r.status, &r.snippet[..40.min(r.snippet.len())])).collect::<Vec<_>>()
    );
    // Active worktree decision should still surface.
    assert!(
        results.iter().any(|r| r.snippet.contains("worktree")),
        "active worktree decision missing: {:?}",
        results.iter().map(|r| &r.snippet).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn comprel_decision_and_procedure_both_retrieved() {
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let query =
        "Why did we decide to vendor kqueue and how do we rebuild the memory crate on OpenBSD?";
    assert!(
        is_compositional_query(query),
        "fixture query must be classified compositional"
    );

    let config = MemorySearchConfig {
        min_score: 0.0,
        max_results: 6,
        ..Default::default()
    };
    let results = compositional_search(&idx, None, query, &config)
        .await
        .unwrap();

    assert!(
        results.requested_kinds.contains(&MemoryKind::Decision),
        "requested {:?}",
        results.requested_kinds
    );
    assert!(
        results.requested_kinds.contains(&MemoryKind::Procedure),
        "requested {:?}",
        results.requested_kinds
    );

    let has_decision = results
        .by_kind
        .get(&MemoryKind::Decision)
        .map(|v| !v.is_empty())
        .unwrap_or(false);
    let has_procedure = results
        .by_kind
        .get(&MemoryKind::Procedure)
        .map(|v| !v.is_empty())
        .unwrap_or(false);

    assert!(
        has_decision,
        "CompRel fail: missing decision hits. flat={:?}",
        results.flat.iter().map(|r| (r.kind, &r.snippet[..80.min(r.snippet.len())])).collect::<Vec<_>>()
    );
    assert!(
        has_procedure,
        "CompRel fail: missing procedure hits. flat={:?}",
        results.flat.iter().map(|r| (r.kind, &r.snippet[..80.min(r.snippet.len())])).collect::<Vec<_>>()
    );

    // Injection template must section both kinds (connector).
    let injection = format_compositional_injection(&results).expect("non-empty injection");
    assert!(injection.contains("Retrieved decisions"));
    assert!(injection.contains("Retrieved procedures"));
    assert!(injection.contains("Prefer retrieved knowledge"));
    assert!(injection.contains("Compose across sections"));
}

#[tokio::test]
async fn comprel_blended_search_can_miss_second_kind() {
    // Documents why typed composition matters: a single blended top-N with
    // max_results=1 can only return one kind, so multi-hop questions fail.
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let query =
        "Why did we decide to vendor kqueue and how do we rebuild the memory crate on OpenBSD?";
    let config = MemorySearchConfig {
        min_score: 0.0,
        max_results: 1,
        ..Default::default()
    };
    let blended = hybrid_search_filtered(
        &idx,
        None,
        query,
        &config,
        &SearchFilter {
            kinds: None,
            exclude_superseded: true,
        },
    )
    .await
    .unwrap();
    assert_eq!(blended.len(), 1, "blended max_results=1 returns one hit");

    // Compositional still aims for multi-kind even with a small budget.
    let mut comp_config = config.clone();
    comp_config.max_results = 4;
    let comp = compositional_search(&idx, None, query, &comp_config)
        .await
        .unwrap();
    let kinds_hit: std::collections::HashSet<_> = comp.flat.iter().map(|r| r.kind).collect();
    assert!(
        kinds_hit.len() >= 2,
        "compositional path should surface >=2 kinds, got {kinds_hit:?}"
    );
}

#[tokio::test]
async fn sequential_edits_preserve_prior_kinds() {
    // Continual-edit analog: append a new fact after initial seed; old
    // decision must still be retrievable (no catastrophic overwrite).
    let tmp = TempDir::new().unwrap();
    let mut idx = test_index(&tmp);
    let path = tmp.path().join("MEMORY.md");
    seed_typed_memory(&mut idx, &path);

    let mut content = std::fs::read_to_string(&path).unwrap();
    content.push_str(
        "\n## Fact: notify uses vendored kqueue\ntype: fact\nstatus: active\n\nnotify links vendor/kqueue on OpenBSD.\n",
    );
    std::fs::write(&path, content).unwrap();
    idx.reindex_file(&path, "workspace").unwrap();

    let config = MemorySearchConfig {
        min_score: 0.0,
        max_results: 10,
        ..Default::default()
    };
    let decisions = hybrid_search_filtered(
        &idx,
        None,
        "vendor kqueue decision",
        &config,
        &SearchFilter {
            kinds: Some(vec![MemoryKind::Decision]),
            exclude_superseded: true,
        },
    )
    .await
    .unwrap();
    assert!(
        !decisions.is_empty(),
        "prior decision must survive sequential fact edit"
    );

    let facts = hybrid_search_filtered(
        &idx,
        None,
        "notify vendored kqueue",
        &config,
        &SearchFilter {
            kinds: Some(vec![MemoryKind::Fact]),
            exclude_superseded: true,
        },
    )
    .await
    .unwrap();
    assert!(!facts.is_empty(), "new fact must be searchable");
}
