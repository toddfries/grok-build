//! SQL schema constants for the memory index.
//!
//! The index uses three tables:
//! - `meta` — key-value metadata (embedding dimensions, schema version)
//! - `chunks` — indexed text chunks with blake3 content hashes
//! - `chunks_fts` — contentless FTS5 virtual table for BM25 keyword search
//!
//! When sqlite-vec is available, a fourth table is created:
//! - `chunks_vec` — vec0 virtual table for KNN vector search
//!
//! ## Schema versioning
//!
//! [`SCHEMA_VERSION`] is stored in `meta`. On open, [`migrate`] applies
//! additive column upgrades for older databases without dropping data.

/// Schema version. Bump when making schema changes that require migration.
///
/// History:
/// - 1: initial chunks + fts + optional vec
/// - 2: typed memory — `kind`, `supersedes`, `status` columns on chunks
pub const SCHEMA_VERSION: u32 = 2;

/// Generate the SQL schema for the memory index.
///
/// `dimensions` controls the embedding vector size for `chunks_vec`.
/// If `vec_available` is false, the `chunks_vec` table is not created.
///
/// Connection pragmas (busy_timeout, journal_mode) are applied on the open
/// path (`xai_sqlite_journal::JournalMode::open`) — the journal mode depends
/// on the database's filesystem.
pub fn schema_sql(dimensions: usize, vec_available: bool) -> String {
    let mut sql = format!(
        r#"
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    hash TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed INTEGER,
    kind TEXT NOT NULL DEFAULT 'unknown',
    supersedes TEXT,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind);
CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='');

INSERT OR IGNORE INTO meta(key, value) VALUES ('reindex_claim', '');
INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION}');
"#
    );

    if vec_available {
        sql.push_str(&format!(
            "\nCREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(\n    \
             chunk_id TEXT PRIMARY KEY,\n    \
             embedding FLOAT[{dimensions}]\n);\n"
        ));
    }

    sql
}

/// Apply additive migrations for databases created before the current schema.
///
/// Safe to call on every open: missing columns are added; existing ones are
/// left alone. Updates `meta.schema_version` when finished.
pub fn migrate(db: &rusqlite::Connection) -> Result<(), rusqlite::Error> {
    let cols = table_columns(db, "chunks")?;
    if !cols.is_empty() {
        // Pre-v2 databases lack typed-memory columns.
        if !cols.contains("kind") {
            db.execute(
                "ALTER TABLE chunks ADD COLUMN kind TEXT NOT NULL DEFAULT 'unknown'",
                [],
            )?;
        }
        if !cols.contains("supersedes") {
            db.execute("ALTER TABLE chunks ADD COLUMN supersedes TEXT", [])?;
        }
        if !cols.contains("status") {
            db.execute(
                "ALTER TABLE chunks ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
                [],
            )?;
        }
        // Indexes are IF NOT EXISTS — cheap to re-issue.
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_kind ON chunks(kind)",
            [],
        )?;
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_status ON chunks(status)",
            [],
        )?;
    }

    db.execute(
        UPSERT_META_SQL,
        rusqlite::params!["schema_version", SCHEMA_VERSION.to_string()],
    )?;
    Ok(())
}

/// Return the set of column names for `table` (empty if the table is missing).
fn table_columns(
    db: &rusqlite::Connection,
    table: &str,
) -> Result<std::collections::HashSet<String>, rusqlite::Error> {
    let mut stmt = db.prepare(&format!("PRAGMA table_info({table})"))?;
    let cols = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    Ok(cols)
}

/// SQL to insert or update an embedding dimension record in the meta table.
pub const UPSERT_META_SQL: &str = "INSERT OR REPLACE INTO meta(key, value) VALUES (?1, ?2)";

/// SQL to query a meta value by key.
pub const GET_META_SQL: &str = "SELECT value FROM meta WHERE key = ?1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_sql_without_vec() {
        let sql = schema_sql(1536, false);
        assert!(sql.contains("CREATE TABLE IF NOT EXISTS chunks"));
        assert!(sql.contains("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts"));
        assert!(sql.contains("kind TEXT NOT NULL DEFAULT 'unknown'"));
        assert!(sql.contains("supersedes TEXT"));
        assert!(sql.contains("status TEXT NOT NULL DEFAULT 'active'"));
        assert!(!sql.contains("chunks_vec"));
        // Connection pragmas live on the open path, not in the schema batch.
        assert!(!sql.contains("PRAGMA"));
    }

    #[test]
    fn test_schema_sql_with_vec() {
        let sql = schema_sql(384, true);
        assert!(sql.contains("chunks_vec"));
        assert!(sql.contains("FLOAT[384]"));
    }

    #[test]
    fn test_schema_sql_different_dimensions() {
        let sql = schema_sql(768, true);
        assert!(sql.contains("FLOAT[768]"));
    }

    #[test]
    fn migrate_adds_typed_columns_to_v1_table() {
        let db = rusqlite::Connection::open_in_memory().unwrap();
        // Simulate a v1 chunks table (no kind/supersedes/status).
        db.execute_batch(
            r#"
            CREATE TABLE chunks (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                hash TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed INTEGER
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            "#,
        )
        .unwrap();
        db.execute(
            "INSERT INTO chunks (id, path, start_line, end_line, text, hash, source, created_at, updated_at)
             VALUES ('a:0', 'MEMORY.md', 0, 1, 'hello', 'h', 'workspace', 0, 0)",
            [],
        )
        .unwrap();

        migrate(&db).unwrap();

        let cols = table_columns(&db, "chunks").unwrap();
        assert!(cols.contains("kind"));
        assert!(cols.contains("supersedes"));
        assert!(cols.contains("status"));

        let kind: String = db
            .query_row("SELECT kind FROM chunks WHERE id = 'a:0'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(kind, "unknown");

        let ver: String = db
            .query_row(GET_META_SQL, ["schema_version"], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, SCHEMA_VERSION.to_string());

        // Idempotent.
        migrate(&db).unwrap();
    }
}
