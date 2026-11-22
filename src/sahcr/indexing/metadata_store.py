"""SQLite metadata storage for indexed code units and AST attributes."""

import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator

from sahcr.models import ExtractedCodeUnit, ParameterInfo


class MetadataStore:
    """Persistent SQLite store for function snippets and structural AST metadata."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS code_units (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    name TEXT NOT NULL,
                    normalized_name_tokens TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    parameter_tokens TEXT NOT NULL,
                    return_annotation TEXT,
                    return_tokens TEXT NOT NULL,
                    docstring TEXT,
                    docstring_tokens TEXT NOT NULL,
                    source_code TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    start_byte INTEGER NOT NULL,
                    end_byte INTEGER NOT NULL,
                    ast_node_types TEXT NOT NULL,
                    is_async BOOLEAN NOT NULL,
                    is_method BOOLEAN NOT NULL,
                    parent_class TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON code_units(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lang ON code_units(language)")
            conn.commit()

    def save_unit(self, unit: ExtractedCodeUnit):
        """Saves or updates a single code unit."""
        self.save_units([unit])

    def save_units(self, units: List[ExtractedCodeUnit]):
        """Bulk saves code units in a single transaction."""
        if not units:
            return

        with self._get_connection() as conn:
            records = []
            for u in units:
                records.append((
                    u.id,
                    u.file_path,
                    u.language,
                    u.name,
                    json.dumps(u.normalized_name_tokens),
                    json.dumps([p.model_dump() for p in u.parameters]),
                    json.dumps(u.parameter_tokens),
                    u.return_annotation,
                    json.dumps(u.return_tokens),
                    u.docstring,
                    json.dumps(u.docstring_tokens),
                    u.source_code,
                    u.start_line,
                    u.end_line,
                    u.start_byte,
                    u.end_byte,
                    json.dumps(u.ast_node_types),
                    1 if u.is_async else 0,
                    1 if u.is_method else 0,
                    u.parent_class
                ))

            conn.executemany("""
                INSERT OR REPLACE INTO code_units (
                    id, file_path, language, name, normalized_name_tokens,
                    parameters_json, parameter_tokens, return_annotation,
                    return_tokens, docstring, docstring_tokens, source_code,
                    start_line, end_line, start_byte, end_byte,
                    ast_node_types, is_async, is_method, parent_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()

    def get_unit(self, unit_id: str) -> Optional[ExtractedCodeUnit]:
        """Retrieves a single code unit by id."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM code_units WHERE id = ?", (unit_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_unit(row)
        return None

    def get_units_by_ids(self, unit_ids: List[str]) -> Dict[str, ExtractedCodeUnit]:
        """Bulk retrieval of units by a list of ids."""
        if not unit_ids:
            return {}

        results = {}
        placeholders = ",".join(["?"] * len(unit_ids))
        with self._get_connection() as conn:
            cursor = conn.execute(f"SELECT * FROM code_units WHERE id IN ({placeholders})", unit_ids)
            for row in cursor.fetchall():
                unit = self._row_to_unit(row)
                results[unit.id] = unit
        return results

    def iter_all_units(self, language: Optional[str] = None) -> Generator[ExtractedCodeUnit, None, None]:
        """Iterates over all stored code units."""
        with self._get_connection() as conn:
            if language:
                cursor = conn.execute("SELECT * FROM code_units WHERE language = ?", (language,))
            else:
                cursor = conn.execute("SELECT * FROM code_units")
            for row in cursor:
                yield self._row_to_unit(row)

    def count(self, language: Optional[str] = None) -> int:
        """Counts total indexed code units."""
        with self._get_connection() as conn:
            if language:
                cursor = conn.execute("SELECT COUNT(*) FROM code_units WHERE language = ?", (language,))
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM code_units")
            return cursor.fetchone()[0]

    def _row_to_unit(self, row: sqlite3.Row) -> ExtractedCodeUnit:
        params_raw = json.loads(row["parameters_json"])
        parameters = [ParameterInfo(**p) for p in params_raw]

        return ExtractedCodeUnit(
            id=row["id"],
            file_path=row["file_path"],
            language=row["language"],
            name=row["name"],
            normalized_name_tokens=json.loads(row["normalized_name_tokens"]),
            parameters=parameters,
            parameter_tokens=json.loads(row["parameter_tokens"]),
            return_annotation=row["return_annotation"],
            return_tokens=json.loads(row["return_tokens"]),
            docstring=row["docstring"],
            docstring_tokens=json.loads(row["docstring_tokens"]),
            source_code=row["source_code"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            start_byte=row["start_byte"],
            end_byte=row["end_byte"],
            ast_node_types=json.loads(row["ast_node_types"]),
            is_async=bool(row["is_async"]),
            is_method=bool(row["is_method"]),
            parent_class=row["parent_class"]
        )

