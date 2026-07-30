//! Format memory search results as `<system-reminder>` content.
//!
//! Used for:
//! - Session start: inject relevant past context on the first turn
//! - Post-compaction: recover relevant memory after context is lost

use xai_chat_state::{MEMORY_CONTEXT_CLOSE_TAG, MEMORY_CONTEXT_OPEN_TAG};
use xai_grok_sampling_types::ConversationItem;
use xai_grok_tools::types::memory_backend::{MemorySearchResult, format_staleness_note};

/// Maximum characters to include per snippet in the injection.
const SNIPPET_MAX_CHARS: usize = 500;

/// Returns `true` if a memory-context block is already persisted in the
/// leading system message. Callers reuse a persisted block verbatim instead
/// of re-searching: a re-scored block would mutate the system-prompt prefix
/// and bust the KV cache for the whole downstream conversation.
pub fn conversation_has_memory_context(items: &[ConversationItem]) -> bool {
    matches!(
        items.first(),
        Some(ConversationItem::System(sys)) if sys.content.contains(MEMORY_CONTEXT_OPEN_TAG)
    )
}

/// Format memory search results as a markdown section for system-reminder injection.
///
/// When results carry typed kinds (Mem-E), groups them into kind sections with
/// a MemEIC-style connector instruction to compose across sections. Otherwise
/// falls back to a flat numbered list (legacy format).
///
/// Returns `None` if results are empty.
pub fn format_memory_reminder(results: &[MemorySearchResult]) -> Option<String> {
    format_memory_reminder_with_core(results, None)
}

/// Like [`format_memory_reminder`], with an optional soft-internal core-pin
/// block (preferences + active decisions) prepended before search hits.
///
/// Returns `None` only when both search results and core pin are empty.
pub fn format_memory_reminder_with_core(
    results: &[MemorySearchResult],
    core_pin: Option<&str>,
) -> Option<String> {
    let core = core_pin.map(str::trim).filter(|s| !s.is_empty());
    if results.is_empty() && core.is_none() {
        return None;
    }

    let mut section = String::from(MEMORY_CONTEXT_OPEN_TAG);
    section.push('\n');

    if let Some(core) = core {
        section.push_str(core);
        section.push_str("\n\n");
    }

    if !results.is_empty() {
        let typed = results
            .iter()
            .any(|r| !r.kind.is_empty() && r.kind != "unknown");
        if typed {
            section.push_str(&format_typed_memory_reminder_body(results));
        } else {
            section.push_str("## Relevant Memory from Past Sessions\n\n");
            for (i, r) in results.iter().enumerate() {
                let truncated = r.snippet.chars().count() > SNIPPET_MAX_CHARS;
                let mut snippet: String = r.snippet.chars().take(SNIPPET_MAX_CHARS).collect();
                if truncated {
                    snippet.push_str("...");
                }
                let staleness = format_staleness_note(&r.source, r.created_at);
                section.push_str(&format!(
                    "### Result {} (score: {:.2}, source: {})\n\
                     **File:** {} (lines {}-{})\n\
                     {}```\n{}\n```\n\n",
                    i + 1,
                    r.score,
                    r.source,
                    r.path,
                    r.start_line,
                    r.end_line,
                    staleness,
                    snippet,
                ));
            }
        }
    }

    section.push_str(MEMORY_CONTEXT_CLOSE_TAG);
    Some(section)
}

/// MemEIC Knowledge Connector analog: section by kind and instruct composition.
///
/// Returns the body (without outer memory-context tags).
fn format_typed_memory_reminder_body(results: &[MemorySearchResult]) -> String {
    use std::collections::BTreeMap;

    let mut by_kind: BTreeMap<&str, Vec<&MemorySearchResult>> = BTreeMap::new();
    for r in results {
        let k = if r.kind.is_empty() {
            "unknown"
        } else {
            r.kind.as_str()
        };
        by_kind.entry(k).or_default().push(r);
    }

    let mut section = String::from(
        "## Relevant Memory from Past Sessions\n\n\
         Prefer retrieved knowledge over model priors when they conflict. \
         Compose across sections when the question needs more than one kind of knowledge.\n\n",
    );

    // Prefer a stable human-useful order rather than pure alphabetical.
    const ORDER: &[&str] = &[
        "decision",
        "fact",
        "procedure",
        "preference",
        "entity",
        "episode",
        "unknown",
    ];
    let mut emitted = std::collections::HashSet::new();
    for kind in ORDER.iter().copied().chain(by_kind.keys().copied()) {
        if !emitted.insert(kind) {
            continue;
        }
        let Some(items) = by_kind.get(kind) else {
            continue;
        };
        section.push_str(&format!("### {}\n\n", kind_section_title(kind)));
        for (i, r) in items.iter().enumerate() {
            let truncated = r.snippet.chars().count() > SNIPPET_MAX_CHARS;
            let mut snippet: String = r.snippet.chars().take(SNIPPET_MAX_CHARS).collect();
            if truncated {
                snippet.push_str("...");
            }
            let staleness = format_staleness_note(&r.source, r.created_at);
            section.push_str(&format!(
                "{}. (score: {:.2}, source: {}, kind: {})\n\
                 **File:** {} (lines {}-{})\n\
                 {}```\n{}\n```\n\n",
                i + 1,
                r.score,
                r.source,
                kind,
                r.path,
                r.start_line,
                r.end_line,
                staleness,
                snippet,
            ));
        }
    }

    section
}

fn kind_section_title(kind: &str) -> &'static str {
    match kind {
        "decision" => "Retrieved decisions",
        "fact" => "Retrieved facts",
        "procedure" => "Retrieved procedures",
        "preference" => "Retrieved preferences",
        "entity" => "Retrieved entities",
        "episode" => "Retrieved session notes",
        _ => "Retrieved memory",
    }
}

/// Whether mid-session Knowledge Connector re-retrieval should run.
///
/// First-turn injection persists core pins + search hits into the system
/// prompt (KV-cache friendly). Mid-session re-query is reserved for
/// **compositional** user turns after that latch has fired, so multi-hop
/// questions re-compose typed stores without rewriting the system prefix.
///
/// Returns `false` for greetings, short text, non-compositional queries, or
/// when first-turn injection has not run yet (first turn owns that path).
pub fn should_mid_session_connector(query: &str, first_turn_done: bool) -> bool {
    if !first_turn_done {
        return false;
    }
    let q = query.trim();
    if q.len() < 20 || is_greeting(q) {
        return false;
    }
    xai_grok_memory::is_compositional_query(q)
}

/// Format a mid-session connector reminder body (no outer memory-context tags).
///
/// Unlike first-turn injection, this is pushed as a `<system-reminder>` so the
/// always-on core-pin block in the system message stays cache-stable.
pub fn format_mid_session_connector_reminder(
    results: &[MemorySearchResult],
) -> Option<String> {
    if results.is_empty() {
        return None;
    }
    let body = format_typed_memory_reminder_body(results);
    Some(format!(
        "Active memory connector (mid-session re-retrieval). \
         Core preferences/decisions in the system prompt still apply. \
         Compose across the sections below for this multi-hop question.\n\n{body}"
    ))
}

/// Check if a message looks like a greeting or generic opener.
///
/// Used to detect vague first messages that won't produce useful memory
/// search results, so we can fall back to a broader project-context query.
pub fn is_greeting(text: &str) -> bool {
    const GREETINGS: &[&str] = &[
        "hi",
        "hey",
        "hello",
        "howdy",
        "continue",
        "start",
        "begin",
        "go",
        "good morning",
        "good afternoon",
        "good evening",
        "what's up",
        "whats up",
        "sup",
    ];
    let lowered = text.to_lowercase();
    let trimmed = lowered.trim().trim_end_matches(['.', '!', '?', ',']);
    GREETINGS.contains(&trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_empty() {
        assert_eq!(format_memory_reminder(&[]), None);
    }

    #[test]
    fn test_format_single_result() {
        let results = vec![MemorySearchResult {
            chunk_id: "test:0".to_string(),
            path: "MEMORY.md".to_string(),
            start_line: 0,
            end_line: 5,
            score: 0.9,
            snippet: "Use tracing for logging, never println!".to_string(),
            source: "workspace".to_string(),
            created_at: None,
            kind: String::new(),
            supersedes: None,
            status: "active".to_string(),
        }];
        let output = format_memory_reminder(&results).unwrap();
        assert!(output.contains("<memory-context>"));
        assert!(output.contains("### Result 1"));
        assert!(output.contains("score: 0.90"));
        assert!(output.contains("**File:** MEMORY.md (lines 0-5)"));
        assert!(output.contains("```\nUse tracing for logging"));
    }

    #[test]
    fn test_format_preserves_newlines() {
        let results = vec![MemorySearchResult {
            chunk_id: "test:0".to_string(),
            path: "MEMORY.md".to_string(),
            start_line: 0,
            end_line: 3,
            score: 0.85,
            snippet: "## Conventions\n\n- Use Rust\n- No clones".to_string(),
            source: "workspace".to_string(),
            created_at: None,
            kind: String::new(),
            supersedes: None,
            status: "active".to_string(),
        }];
        let output = format_memory_reminder(&results).unwrap();
        assert!(
            output.contains("## Conventions\n\n- Use Rust\n- No clones"),
            "newlines in snippet should be preserved, not collapsed"
        );
    }

    #[test]
    fn test_format_truncates_long_snippets() {
        let results = vec![MemorySearchResult {
            chunk_id: "test:0".to_string(),
            path: "test.md".to_string(),
            start_line: 0,
            end_line: 5,
            score: 0.8,
            snippet: "x".repeat(1000),
            source: "session".to_string(),
            created_at: None,
            kind: String::new(),
            supersedes: None,
            status: "active".to_string(),
        }];
        let output = format_memory_reminder(&results).unwrap();
        // Snippet should be truncated to SNIPPET_MAX_CHARS (500) + "..."
        assert!(!output.contains(&"x".repeat(501)));
        assert!(output.contains(&format!("{}...", "x".repeat(500))));
    }

    #[test]
    fn test_format_multiple_results() {
        let results = vec![
            MemorySearchResult {
                chunk_id: "a:0".to_string(),
                path: "MEMORY.md".to_string(),
                start_line: 0,
                end_line: 5,
                score: 0.9,
                snippet: "First result".to_string(),
                source: "workspace".to_string(),
                created_at: None,
                kind: String::new(),
                supersedes: None,
                status: "active".to_string(),
            },
            MemorySearchResult {
                chunk_id: "b:0".to_string(),
                path: "session.md".to_string(),
                start_line: 10,
                end_line: 15,
                score: 0.7,
                snippet: "Second result".to_string(),
                source: "session".to_string(),
                created_at: None,
                kind: String::new(),
                supersedes: None,
                status: "active".to_string(),
            },
        ];
        let output = format_memory_reminder(&results).unwrap();
        assert!(output.contains("### Result 1"));
        assert!(output.contains("### Result 2"));
        assert!(output.contains("score: 0.90"));
        assert!(output.contains("score: 0.70"));
    }

    // -----------------------------------------------------------------------
    // conversation_has_memory_context (idempotency guard) tests
    // -----------------------------------------------------------------------

    fn sample_result() -> MemorySearchResult {
        MemorySearchResult {
            chunk_id: "test:0".into(),
            path: "MEMORY.md".into(),
            start_line: 0,
            end_line: 5,
            score: 0.9,
            snippet: "Project uses Rust for backend services.".into(),
            source: "workspace".into(),
            created_at: None,
            kind: String::new(),
            supersedes: None,
            status: "active".to_string(),
        }
    }

    #[test]
    fn test_detects_persisted_block_in_system_message() {
        let block = format_memory_reminder(&[sample_result()]).unwrap();
        let system_content = format!("You are a helpful assistant.\n\n{block}");
        let conversation = vec![
            ConversationItem::system(system_content),
            ConversationItem::user("help me fix the auth bug"),
        ];
        assert!(
            conversation_has_memory_context(&conversation),
            "an already-injected memory-context block must be detected so it is reused, not re-searched"
        );
    }

    #[test]
    fn test_no_block_when_system_lacks_marker() {
        let conversation = vec![
            ConversationItem::system("You are a helpful assistant."),
            ConversationItem::user("hi"),
        ];
        assert!(!conversation_has_memory_context(&conversation));
    }

    #[test]
    fn test_no_block_when_no_leading_system_message() {
        let conversation = vec![ConversationItem::user("hi")];
        assert!(!conversation_has_memory_context(&conversation));
    }

    #[test]
    fn test_no_block_for_empty_conversation() {
        assert!(!conversation_has_memory_context(&[]));
    }

    // -----------------------------------------------------------------------
    // staleness annotation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_staleness_shown_for_old_session_result() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let results = vec![MemorySearchResult {
            chunk_id: "s:0".into(),
            path: "session.md".into(),
            start_line: 0,
            end_line: 5,
            score: 0.8,
            snippet: "old info".into(),
            source: "session".into(),
            created_at: Some(now - 86400 * 10),
        kind: String::new(),
        supersedes: None,
        status: "active".to_string(),
        }];
        let output = format_memory_reminder(&results).unwrap();
        assert!(
            output.contains("**Stale ("),
            "10-day-old session result should show stale warning, got: {output}"
        );
    }

    #[test]
    fn test_no_staleness_for_workspace_result() {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let results = vec![MemorySearchResult {
            chunk_id: "w:0".into(),
            path: "MEMORY.md".into(),
            start_line: 0,
            end_line: 5,
            score: 0.9,
            snippet: "workspace data".into(),
            source: "workspace".into(),
            created_at: Some(now - 86400 * 30),
        kind: String::new(),
        supersedes: None,
        status: "active".to_string(),
        }];
        let output = format_memory_reminder(&results).unwrap();
        assert!(
            !output.contains("**Stale (") && !output.contains("**Note ("),
            "workspace result must not show staleness, got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // is_greeting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_greeting_detection() {
        assert!(is_greeting("hi"));
        assert!(is_greeting("Hey!"));
        assert!(is_greeting("Hello."));
        assert!(is_greeting("good morning"));
        assert!(is_greeting("continue"));
        assert!(is_greeting("  HELLO  "));
    }

    #[test]
    fn test_non_greeting() {
        assert!(!is_greeting("help me fix the auth bug"));
        assert!(!is_greeting("implement feature X"));
        assert!(!is_greeting("what does this function do"));
        assert!(!is_greeting("hi there, can you help me with something"));
    }

    // -----------------------------------------------------------------------
    // Injection counter semantics tests
    // -----------------------------------------------------------------------

    /// `format_memory_reminder` returns `None` for an empty result list.
    ///
    /// This is the key invariant for the `memory_injection_count` contract:
    /// the counter must only be incremented when `memory_reminder.is_some()`,
    /// which is only true when `format_memory_reminder` returns `Some(_)`.
    /// An empty result set must produce `None`, preventing the counter from
    /// overcounting attempts where memory search found nothing to inject.
    #[test]
    fn test_format_memory_reminder_empty_results_is_none() {
        use xai_grok_tools::types::memory_backend::MemorySearchResult;
        let results: Vec<MemorySearchResult> = vec![];
        let reminder = format_memory_reminder(&results);
        assert!(
            reminder.is_none(),
            "empty results must produce None — injection_count must NOT increment"
        );
    }

    /// `format_memory_reminder` returns `Some(_)` for a non-empty result list.
    ///
    /// Confirms that `memory_injection_count` correctly increments when there
    /// are actual results to inject.
    #[test]
    fn test_format_memory_reminder_with_results_is_some() {
        use xai_grok_tools::types::memory_backend::MemorySearchResult;
        let results = vec![MemorySearchResult {
            chunk_id: "test:0".into(),
            path: "/mem/MEMORY.md".into(),
            start_line: 0,
            end_line: 3,
            score: 0.85,
            snippet: "Project uses Rust for backend services.".into(),
            source: "workspace".into(),
            created_at: None,
            kind: String::new(),
            supersedes: None,
            status: "active".to_string(),
        }];
        let reminder = format_memory_reminder(&results);
        assert!(
            reminder.is_some(),
            "non-empty results must produce Some(_) — injection_count SHOULD increment"
        );
    }

    // -----------------------------------------------------------------------
    // mid-session connector policy
    // -----------------------------------------------------------------------

    #[test]
    fn mid_session_skips_until_first_turn_done() {
        assert!(!should_mid_session_connector(
            "Why did we choose worktrees and how do we rebuild?",
            false
        ));
    }

    #[test]
    fn mid_session_fires_on_compositional_after_first_turn() {
        assert!(should_mid_session_connector(
            "Why did we choose worktrees and how do we rebuild on OpenBSD?",
            true
        ));
    }

    #[test]
    fn mid_session_skips_greetings_and_short() {
        assert!(!should_mid_session_connector("hi", true));
        assert!(!should_mid_session_connector("fix flaky test", true));
    }

    #[test]
    fn mid_session_reminder_wraps_typed_body() {
        let results = vec![MemorySearchResult {
            chunk_id: "a:0".into(),
            path: "MEMORY.md".into(),
            start_line: 0,
            end_line: 2,
            score: 0.9,
            snippet: "Use worktrees.".into(),
            source: "workspace".into(),
            created_at: None,
            kind: "decision".into(),
            supersedes: None,
            status: "active".into(),
        }];
        let out = format_mid_session_connector_reminder(&results).unwrap();
        assert!(out.contains("Active memory connector"));
        assert!(out.contains("Retrieved decisions") || out.contains("decision"));
        assert!(out.contains("Use worktrees"));
        assert!(!out.contains("<memory-context>"));
    }

    #[test]
    fn mid_session_reminder_empty_is_none() {
        assert!(format_mid_session_connector_reminder(&[]).is_none());
    }
}
