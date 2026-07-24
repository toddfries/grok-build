//! Typed memory kinds for MemEIC-style dual/multi-store retrieval.
//!
//! Knowledge types that interfere when mixed in a single bag-of-results
//! (decisions vs facts vs procedures) are classified at index time from
//! markdown structure and optional front-matter fields:
//!
//! ```markdown
//! ## Decision: use worktree isolation for execute-plan
//! type: decision
//! status: active
//! supersedes: 2026-03-01-shared-cwd
//!
//! We isolate execute-plan subagents in git worktrees so parent state is safe.
//! ```
//!
//! Heading prefixes (`## Decision:`, `## Fact:`, …) are also recognized when
//! no explicit `type:` field is present. Session-sourced chunks without an
//! explicit type default to [`MemoryKind::Episode`].

use std::fmt;
use std::str::FromStr;

/// Knowledge type for a memory chunk.
///
/// Used to partition external memory (Mem-E) so compositional queries can
/// retrieve best-of-kind rather than top-N of a blended score soup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryKind {
    /// Durable factual statements ("X is true of this repo").
    Fact,
    /// Architectural / product decisions with rationale.
    Decision,
    /// User or project preferences and habits.
    Preference,
    /// How-to / runbook / debugging procedure.
    Procedure,
    /// Session-scoped episodic notes (decaying).
    Episode,
    /// Named entities (people, services, tickets) — optional store.
    Entity,
    /// Unclassified; treated as eligible for any untyped search.
    Unknown,
}

impl MemoryKind {
    /// Stable wire / SQL string for this kind.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Procedure => "procedure",
            Self::Episode => "episode",
            Self::Entity => "entity",
            Self::Unknown => "unknown",
        }
    }

    /// Section title used in compositional injection templates.
    pub fn section_title(self) -> &'static str {
        match self {
            Self::Fact => "Retrieved facts",
            Self::Decision => "Retrieved decisions",
            Self::Preference => "Retrieved preferences",
            Self::Procedure => "Retrieved procedures",
            Self::Episode => "Retrieved session notes",
            Self::Entity => "Retrieved entities",
            Self::Unknown => "Retrieved memory",
        }
    }

    /// All concrete kinds (excludes [`Self::Unknown`]).
    pub fn all_typed() -> &'static [MemoryKind] {
        &[
            Self::Fact,
            Self::Decision,
            Self::Preference,
            Self::Procedure,
            Self::Episode,
            Self::Entity,
        ]
    }
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for MemoryKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.trim().to_ascii_lowercase().as_str() {
            "fact" | "facts" => Self::Fact,
            "decision" | "decisions" | "adr" => Self::Decision,
            "preference" | "preferences" | "pref" | "prefs" => Self::Preference,
            "procedure" | "procedures" | "runbook" | "howto" | "how-to" => Self::Procedure,
            "episode" | "session" | "sessions" => Self::Episode,
            "entity" | "entities" => Self::Entity,
            "unknown" | "" => Self::Unknown,
            _ => return Err(()),
        })
    }
}

/// Metadata extracted from a memory chunk at index time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkMeta {
    pub kind: MemoryKind,
    /// Optional id / slug of a chunk this entry supersedes.
    pub supersedes: Option<String>,
    /// `active` (default) or `superseded`.
    pub status: String,
}

impl Default for ChunkMeta {
    fn default() -> Self {
        Self {
            kind: MemoryKind::Unknown,
            supersedes: None,
            status: "active".to_string(),
        }
    }
}

/// Classify a chunk's text given its storage `source` (`global` / `workspace` / `session`).
pub fn classify_chunk(text: &str, source: &str) -> ChunkMeta {
    let mut meta = ChunkMeta::default();

    // Scan the first ~40 non-empty lines for fields / heading cues.
    let mut scanned = 0usize;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        scanned += 1;
        if scanned > 40 {
            break;
        }

        if let Some(value) = field_value(trimmed, "type") {
            if let Ok(k) = MemoryKind::from_str(value) {
                meta.kind = k;
            }
        } else if let Some(value) = field_value(trimmed, "kind") {
            if let Ok(k) = MemoryKind::from_str(value) {
                meta.kind = k;
            }
        } else if let Some(value) = field_value(trimmed, "status") {
            let v = value.trim().to_ascii_lowercase();
            if matches!(v.as_str(), "active" | "superseded" | "deprecated") {
                meta.status = v;
            }
        } else if let Some(value) = field_value(trimmed, "supersedes") {
            let v = value.trim();
            if !v.is_empty() {
                meta.supersedes = Some(v.to_string());
            }
        } else if meta.kind == MemoryKind::Unknown {
            if let Some(k) = kind_from_heading(trimmed) {
                meta.kind = k;
            }
        }
    }

    if meta.kind == MemoryKind::Unknown && source == "session" {
        meta.kind = MemoryKind::Episode;
    }

    meta
}

/// `key: value` field parser (case-insensitive key, no YAML complexity).
fn field_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let (left, right) = line.split_once(':')?;
    if !left.trim().eq_ignore_ascii_case(key) {
        return None;
    }
    // Avoid matching markdown headings like "## Type: Foo" — those go through
    // kind_from_heading. Field keys must not start with '#'.
    if left.trim_start().starts_with('#') {
        return None;
    }
    Some(right.trim())
}

/// Recognize `## Decision: …`, `### Fact — …`, `## Preferences`, etc.
fn kind_from_heading(line: &str) -> Option<MemoryKind> {
    let trimmed = line.trim();
    if !trimmed.starts_with('#') {
        return None;
    }
    let title = trimmed
        .trim_start_matches('#')
        .trim()
        .to_ascii_lowercase();
    // Strip common separators after the kind word.
    let word = title
        .split(|c: char| c == ':' || c == '—' || c == '-' || c == '|' || c.is_whitespace())
        .find(|w| !w.is_empty())?;

    match word {
        "decision" | "decisions" | "adr" => Some(MemoryKind::Decision),
        "fact" | "facts" => Some(MemoryKind::Fact),
        "preference" | "preferences" | "pref" => Some(MemoryKind::Preference),
        "procedure" | "procedures" | "runbook" | "howto" => Some(MemoryKind::Procedure),
        "episode" | "session" | "sessions" => Some(MemoryKind::Episode),
        "entity" | "entities" => Some(MemoryKind::Entity),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_type_field() {
        let text = "## Something\ntype: decision\nstatus: active\n\nWe chose X.\n";
        let meta = classify_chunk(text, "workspace");
        assert_eq!(meta.kind, MemoryKind::Decision);
        assert_eq!(meta.status, "active");
    }

    #[test]
    fn classifies_heading_prefix() {
        let text = "## Decision: use worktrees\n\nIsolate subagents.\n";
        let meta = classify_chunk(text, "workspace");
        assert_eq!(meta.kind, MemoryKind::Decision);
    }

    #[test]
    fn classifies_supersedes() {
        let text = "## Fact: API version\ntype: fact\nsupersedes: old-api-v1\n\nAPI is v2.\n";
        let meta = classify_chunk(text, "workspace");
        assert_eq!(meta.kind, MemoryKind::Fact);
        assert_eq!(meta.supersedes.as_deref(), Some("old-api-v1"));
    }

    #[test]
    fn session_defaults_to_episode() {
        let text = "Discussed the flaky test briefly.\n";
        let meta = classify_chunk(text, "session");
        assert_eq!(meta.kind, MemoryKind::Episode);
    }

    #[test]
    fn workspace_unknown_without_cues() {
        let text = "Some free-form notes without structure.\n";
        let meta = classify_chunk(text, "workspace");
        assert_eq!(meta.kind, MemoryKind::Unknown);
    }

    #[test]
    fn preference_heading() {
        let text = "## Preferences\n\nAlways open PR links after push.\n";
        let meta = classify_chunk(text, "global");
        assert_eq!(meta.kind, MemoryKind::Preference);
    }

    #[test]
    fn procedure_heading() {
        let text = "### Procedure: rebuild on OpenBSD\n\n1. cargo build -p xai-grok-memory\n";
        let meta = classify_chunk(text, "workspace");
        assert_eq!(meta.kind, MemoryKind::Procedure);
    }

    #[test]
    fn parse_aliases() {
        assert_eq!(MemoryKind::from_str("adr").unwrap(), MemoryKind::Decision);
        assert_eq!(MemoryKind::from_str("runbook").unwrap(), MemoryKind::Procedure);
        assert!(MemoryKind::from_str("nonsense").is_err());
    }
}
