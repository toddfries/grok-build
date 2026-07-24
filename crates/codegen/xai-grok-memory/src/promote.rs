//! Soft Mem-I promotion ladder: episode → typed evergreen → core pin.
//!
//! MemEIC internalizes edits into modality-specific adapters. Grok Build
//! cannot edit hosted-model weights, so "internalization" is a promotion
//! ladder that moves knowledge into progressively more durable, always-
//! available surfaces:
//!
//! ```text
//! session log  →  (flush/dream)  →  typed evergreen MEMORY.md
//!                                    ↓ core-pin selection
//!                              always-inject soft-internal core
//! ```
//!
//! This module formats typed appends (with supersession metadata) and
//! extracts a bounded **core pin** set of preferences + active decisions
//! for first-turn injection without a search call.

use super::kind::MemoryKind;
use super::storage::{MemoryScope, MemoryStorage, normalize_memory_content};

/// Default character budget for always-on soft-internal memory.
pub const DEFAULT_CORE_PIN_MAX_CHARS: usize = 2_000;

/// Maximum number of core-pin sections (after char budget).
pub const DEFAULT_CORE_PIN_MAX_SECTIONS: usize = 8;

/// Options for selecting the soft-internal core pin set.
#[derive(Debug, Clone)]
pub struct CorePinConfig {
    pub max_chars: usize,
    pub max_sections: usize,
    /// Kinds eligible for always-on injection, in priority order.
    pub kind_priority: Vec<MemoryKind>,
}

impl Default for CorePinConfig {
    fn default() -> Self {
        Self {
            max_chars: DEFAULT_CORE_PIN_MAX_CHARS,
            max_sections: DEFAULT_CORE_PIN_MAX_SECTIONS,
            kind_priority: vec![
                MemoryKind::Preference,
                MemoryKind::Decision,
                MemoryKind::Fact,
            ],
        }
    }
}

/// A single typed memory entry ready to append to evergreen storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedEntry {
    pub kind: MemoryKind,
    pub title: String,
    pub body: String,
    pub status: String,
    pub supersedes: Option<String>,
    pub id: Option<String>,
}

impl TypedEntry {
    /// Build a new active entry.
    pub fn new(kind: MemoryKind, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            body: body.into(),
            status: "active".to_string(),
            supersedes: None,
            id: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn superseding(mut self, old_id: impl Into<String>) -> Self {
        self.supersedes = Some(old_id.into());
        self
    }

    pub fn superseded(mut self) -> Self {
        self.status = "superseded".to_string();
        self
    }

    /// Render as a markdown section with type/status/supersedes fields.
    pub fn to_markdown(&self) -> String {
        let kind_label = match self.kind {
            MemoryKind::Decision => "Decision",
            MemoryKind::Fact => "Fact",
            MemoryKind::Preference => "Preference",
            MemoryKind::Procedure => "Procedure",
            MemoryKind::Episode => "Episode",
            MemoryKind::Entity => "Entity",
            MemoryKind::Unknown => "Note",
        };
        let mut out = format!("## {kind_label}: {}\n", self.title.trim());
        out.push_str(&format!("type: {}\n", self.kind.as_str()));
        out.push_str(&format!("status: {}\n", self.status));
        if let Some(id) = &self.id {
            out.push_str(&format!("id: {id}\n"));
        }
        if let Some(s) = &self.supersedes {
            out.push_str(&format!("supersedes: {s}\n"));
        }
        out.push('\n');
        let body = self.body.trim();
        if !body.is_empty() {
            out.push_str(body);
            out.push('\n');
        }
        out
    }
}

/// Append a typed entry to global or workspace MEMORY.md.
pub fn promote_entry(
    storage: &MemoryStorage,
    scope: MemoryScope,
    entry: &TypedEntry,
) -> std::io::Result<()> {
    let md = entry.to_markdown();
    storage.append_to_memory(scope, &md)
}

/// Format free-form "remember" text as a typed evergreen section.
///
/// Heuristically picks kind from the note text when not specified.
pub fn format_remember_note(note: &str, kind: Option<MemoryKind>) -> String {
    let note = note.trim();
    if note.is_empty() {
        return String::new();
    }
    let kind = kind.unwrap_or_else(|| infer_kind_from_note(note));
    let title = title_from_note(note);
    TypedEntry::new(kind, title, note.to_string()).to_markdown()
}

fn infer_kind_from_note(note: &str) -> MemoryKind {
    let lower = note.to_ascii_lowercase();
    if lower.contains("decid") || lower.contains("we chose") || lower.contains("architecture") {
        MemoryKind::Decision
    } else if lower.contains("always")
        || lower.contains("never ")
        || lower.contains("prefer")
        || lower.contains("preference")
    {
        MemoryKind::Preference
    } else if lower.contains("how to")
        || lower.contains("steps:")
        || lower.contains("runbook")
        || lower.contains("procedure")
    {
        MemoryKind::Procedure
    } else {
        MemoryKind::Fact
    }
}

fn title_from_note(note: &str) -> String {
    let first = note.lines().next().unwrap_or(note).trim();
    let stripped = first
        .trim_start_matches('#')
        .trim()
        .trim_start_matches("remember")
        .trim_start_matches("Remember")
        .trim_start_matches(':')
        .trim();
    let mut title: String = stripped.chars().take(80).collect();
    if stripped.chars().count() > 80 {
        title.push('…');
    }
    if title.is_empty() {
        "note".to_string()
    } else {
        title
    }
}

/// A core-pin section extracted from evergreen MEMORY.md content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorePinSection {
    pub kind: MemoryKind,
    pub title: String,
    pub text: String,
}

/// Extract a bounded soft-internal core from evergreen markdown.
///
/// Only **active** preference / decision / fact sections (per config priority)
/// are included, subject to char and section caps. Superseded sections are
/// skipped. This is the agent analog of Mem-I "always-on" adapters.
pub fn extract_core_pins(markdown: &str, config: &CorePinConfig) -> Vec<CorePinSection> {
    let sections = split_h2_sections(markdown);
    let mut candidates: Vec<(usize, CorePinSection)> = Vec::new();

    for (title, body) in sections {
        let full = format!("## {title}\n{body}");
        let meta = super::kind::classify_chunk(&full, "workspace");
        if meta.status == "superseded" || meta.status == "deprecated" {
            continue;
        }
        let kind = meta.kind;
        let Some(priority) = config.kind_priority.iter().position(|k| *k == kind) else {
            continue;
        };
        candidates.push((
            priority,
            CorePinSection {
                kind,
                title,
                text: full.trim().to_string(),
            },
        ));
    }

    // Sort by priority (lower index first), then keep within budgets.
    candidates.sort_by_key(|(p, _)| *p);

    let mut selected = Vec::new();
    let mut used_chars = 0usize;
    for (_, pin) in candidates {
        if selected.len() >= config.max_sections {
            break;
        }
        if pin.text.is_empty() {
            continue;
        }
        if used_chars + pin.text.len() > config.max_chars && !selected.is_empty() {
            continue;
        }
        used_chars += pin.text.len();
        selected.push(pin);
    }
    selected
}

/// Format core pins for system-prompt injection (no search required).
pub fn format_core_pin_injection(pins: &[CorePinSection]) -> Option<String> {
    if pins.is_empty() {
        return None;
    }
    let mut out = String::from(
        "## Soft-Internal Core (always on)\n\n\
         These are high-priority preferences and active decisions. \
         Prefer them over model priors. They do not expire with session logs.\n\n",
    );
    for pin in pins {
        out.push_str(&pin.text);
        if !pin.text.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }
    Some(out)
}

/// Load core pins from global + workspace MEMORY.md files.
pub fn load_core_pins(storage: &MemoryStorage, config: &CorePinConfig) -> Vec<CorePinSection> {
    let mut all_md = String::new();
    for path in [storage.global_memory_file(), storage.workspace_memory_file()] {
        if !path.exists() {
            continue;
        }
        if let Ok(content) = std::fs::read_to_string(&path) {
            if !content.trim().is_empty() {
                if !all_md.is_empty() {
                    all_md.push_str("\n\n");
                }
                all_md.push_str(&content);
            }
        }
    }
    if all_md.is_empty() {
        return Vec::new();
    }
    extract_core_pins(&all_md, config)
}

/// Mark an existing section as superseded in-place (best-effort string rewrite).
///
/// Looks for `id: {id}` blocks and sets `status: superseded`.
/// Returns `Some(rewritten)` if a change was made.
pub fn mark_superseded_in_markdown(markdown: &str, id: &str) -> Option<String> {
    if id.is_empty() || !markdown.contains(id) {
        return None;
    }

    let sections = split_h2_sections_raw(markdown);
    let mut changed = false;
    let mut out = String::new();

    // Preserve any preamble before the first ##.
    if let Some(preamble) = preamble_before_h2(markdown) {
        out.push_str(preamble);
        if !preamble.ends_with('\n') {
            out.push('\n');
        }
    }

    for (heading_line, body) in sections {
        let full_for_id = format!("{heading_line}\n{body}");
        let is_target = full_for_id.lines().any(|l| {
            l.trim()
                .strip_prefix("id:")
                .is_some_and(|rest| rest.trim() == id)
                || heading_line.contains(id)
        });

        out.push_str(&heading_line);
        out.push('\n');

        if is_target {
            let mut saw_status = false;
            for line in body.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("status:") {
                    out.push_str("status: superseded\n");
                    saw_status = true;
                    changed = true;
                } else {
                    out.push_str(line);
                    out.push('\n');
                }
            }
            if !saw_status {
                out.push_str("status: superseded\n");
                changed = true;
            }
        } else {
            out.push_str(&body);
            if !body.is_empty() && !body.ends_with('\n') {
                out.push('\n');
            }
        }
        out.push('\n');
    }

    if changed {
        Some(out)
    } else {
        None
    }
}

/// Promote a note: optionally mark an old id superseded, then append the new entry.
pub fn promote_with_supersession(
    storage: &MemoryStorage,
    scope: MemoryScope,
    entry: &TypedEntry,
) -> std::io::Result<()> {
    if let Some(old_id) = &entry.supersedes {
        let path = match scope {
            MemoryScope::Global => storage.global_memory_file(),
            MemoryScope::Workspace => storage.workspace_memory_file(),
        };
        if path.exists() {
            if let Ok(existing) = std::fs::read_to_string(&path) {
                if let Some(rewritten) = mark_superseded_in_markdown(&existing, old_id) {
                    storage.write_long_term(scope, &rewritten)?;
                }
            }
        }
    }
    promote_entry(storage, scope, entry)
}

/// Ensure dream / flush output keeps usable structure.
pub fn ensure_typed_structure(content: &str) -> String {
    normalize_memory_content(content)
}

/// Split markdown into `(heading_title, body)` on `##` headers.
fn split_h2_sections(markdown: &str) -> Vec<(String, String)> {
    split_h2_sections_raw(markdown)
        .into_iter()
        .map(|(heading, body)| {
            let title = heading.trim_start_matches('#').trim().to_string();
            (title, body)
        })
        .collect()
}

/// Split into `(full_heading_line, body)` pairs.
fn split_h2_sections_raw(markdown: &str) -> Vec<(String, String)> {
    let mut sections = Vec::new();
    let mut current_heading: Option<String> = None;
    let mut body = String::new();

    for line in markdown.lines() {
        if is_h2(line) {
            if let Some(h) = current_heading.take() {
                sections.push((h, std::mem::take(&mut body)));
            }
            current_heading = Some(line.to_string());
        } else if current_heading.is_some() {
            body.push_str(line);
            body.push('\n');
        }
    }
    if let Some(h) = current_heading {
        sections.push((h, body));
    }
    sections
}

fn is_h2(line: &str) -> bool {
    let t = line.trim_start();
    t.starts_with("## ") && !t.starts_with("### ")
}

fn preamble_before_h2(markdown: &str) -> Option<&str> {
    let idx = markdown.find("\n## ")?;
    let pre = &markdown[..idx + 1];
    if pre.trim().is_empty() {
        None
    } else {
        Some(pre)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemoryStorage;
    use tempfile::TempDir;

    #[test]
    fn typed_entry_markdown_has_fields() {
        let e = TypedEntry::new(MemoryKind::Decision, "use worktrees", "Isolate subagents.")
            .with_id("dec-worktree")
            .superseding("dec-shared-cwd");
        let md = e.to_markdown();
        assert!(md.contains("## Decision: use worktrees"));
        assert!(md.contains("type: decision"));
        assert!(md.contains("status: active"));
        assert!(md.contains("id: dec-worktree"));
        assert!(md.contains("supersedes: dec-shared-cwd"));
        assert!(md.contains("Isolate subagents."));
    }

    #[test]
    fn format_remember_infers_preference() {
        let md = format_remember_note("always open PR links after push", None);
        assert!(md.contains("type: preference"));
    }

    #[test]
    fn format_remember_infers_decision() {
        let md = format_remember_note("we decided to vendor kqueue for OpenBSD", None);
        assert!(md.contains("type: decision"));
    }

    #[test]
    fn extract_core_pins_prefers_active_preferences_and_decisions() {
        let md = r#"# Memory

## Preference: commit after green
type: preference
status: active

Always commit when tests pass.

## Decision: shared cwd
type: decision
status: superseded

Old approach.

## Decision: worktree isolation
type: decision
status: active

Use worktrees.

## Procedure: rebuild
type: procedure
status: active

cargo test -p xai-grok-memory
"#;
        let pins = extract_core_pins(md, &CorePinConfig::default());
        assert!(
            pins.iter().any(|p| p.kind == MemoryKind::Preference),
            "expected preference pin: {pins:?}"
        );
        assert!(
            pins.iter()
                .any(|p| p.kind == MemoryKind::Decision && p.text.contains("worktree")),
            "expected active decision: {pins:?}"
        );
        assert!(
            !pins.iter().any(|p| p.text.contains("shared cwd")),
            "superseded decision should not be pinned: {pins:?}"
        );
        assert!(
            pins.iter().all(|p| p.kind != MemoryKind::Procedure),
            "procedures should not be core-pinned by default: {pins:?}"
        );
        // Preference before decision (priority order).
        let pref_i = pins.iter().position(|p| p.kind == MemoryKind::Preference);
        let dec_i = pins.iter().position(|p| p.kind == MemoryKind::Decision);
        if let (Some(p), Some(d)) = (pref_i, dec_i) {
            assert!(p < d, "preferences should rank before decisions");
        }
    }

    #[test]
    fn core_pin_respects_char_budget() {
        let mut md = String::from("# Memory\n\n");
        for i in 0..20 {
            md.push_str(&format!(
                "## Preference: pref {i}\ntype: preference\nstatus: active\n\n{}\n\n",
                "x".repeat(200)
            ));
        }
        let config = CorePinConfig {
            max_chars: 500,
            max_sections: 20,
            ..Default::default()
        };
        let pins = extract_core_pins(&md, &config);
        let total: usize = pins.iter().map(|p| p.text.len()).sum();
        assert!(total <= 500 + 300, "budget roughly respected: {total}");
        assert!(!pins.is_empty());
    }

    #[test]
    fn mark_superseded_rewrites_status() {
        let md = "# Memory\n\n## Decision: old\ntype: decision\nstatus: active\nid: old-dec\n\nBody.\n";
        let out = mark_superseded_in_markdown(md, "old-dec").unwrap();
        assert!(out.contains("status: superseded"));
        assert!(!out.contains("status: active"));
    }

    #[test]
    fn promote_entry_appends_to_workspace() {
        let tmp = TempDir::new().unwrap();
        let global = tmp.path().join("memory");
        let workspace = global.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let storage = MemoryStorage::with_paths(global, workspace);
        let entry = TypedEntry::new(MemoryKind::Fact, "schema v2", "schema_version is 2");
        promote_entry(&storage, MemoryScope::Workspace, &entry).unwrap();
        let content = std::fs::read_to_string(storage.workspace_memory_file()).unwrap();
        assert!(content.contains("type: fact"));
        assert!(content.contains("schema_version is 2"));
    }

    #[test]
    fn format_core_pin_injection_none_when_empty() {
        assert!(format_core_pin_injection(&[]).is_none());
    }

    #[test]
    fn format_core_pin_injection_has_header() {
        let pins = vec![CorePinSection {
            kind: MemoryKind::Preference,
            title: "commit".into(),
            text: "## Preference: commit\ntype: preference\nstatus: active\n\nAlways commit.\n"
                .into(),
        }];
        let s = format_core_pin_injection(&pins).unwrap();
        assert!(s.contains("Soft-Internal Core"));
        assert!(s.contains("Always commit"));
    }

    #[test]
    fn promote_with_supersession_marks_old() {
        let tmp = TempDir::new().unwrap();
        let global = tmp.path().join("memory");
        let workspace = global.join("ws");
        std::fs::create_dir_all(&workspace).unwrap();
        let storage = MemoryStorage::with_paths(global, workspace);
        let old = TypedEntry::new(MemoryKind::Decision, "shared cwd", "Use parent cwd.")
            .with_id("dec-shared");
        promote_entry(&storage, MemoryScope::Workspace, &old).unwrap();
        let new = TypedEntry::new(MemoryKind::Decision, "worktrees", "Use worktrees.")
            .with_id("dec-wt")
            .superseding("dec-shared");
        promote_with_supersession(&storage, MemoryScope::Workspace, &new).unwrap();
        let content = std::fs::read_to_string(storage.workspace_memory_file()).unwrap();
        assert!(content.contains("status: superseded"));
        assert!(content.contains("worktrees"));
        assert!(content.contains("supersedes: dec-shared"));
    }
}
