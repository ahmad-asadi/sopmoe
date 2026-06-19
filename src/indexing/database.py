"""Vector database wrapper using FAISS + SQLite for the offline index.

Stores embedding vectors in a FAISS flat L2 index and metadata
(expert_id, timestamp, SoP text, performance metrics) in SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    """FAISS flat-L2 index paired with a SQLite metadata store.

    Each record in the index stores:
      - embedding ``h_τ`` (float32) in FAISS
      - metadata (expert_id, timestep, SoP text, performance metrics) in SQLite

    When no ``db_path`` is provided at construction, an in-memory SQLite
    database is used (data is lost on close unless explicitly saved).
    """

    def __init__(
        self,
        dim: int,
        index_path: str | Path | None = None,
        db_path: str | Path | None = None,
    ):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self._db_path: Path | None = Path(db_path) if db_path else None
        self.con: sqlite3.Connection | None = None
        self._next_id: int = 0

        # Always open an in-memory or file-backed SQLite DB
        self._open_db(self._db_path)

        if index_path is not None:
            self._load_index(Path(index_path))

    # ------------------------------------------------------------------
    # SQLite metadata
    # ------------------------------------------------------------------

    def _open_db(self, path: Path | None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.con = sqlite3.connect(str(path))
        else:
            self.con = sqlite3.connect(":memory:")
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS index_records (
                id                 INTEGER PRIMARY KEY,
                faiss_id           INTEGER NOT NULL,
                timestamp          TEXT NOT NULL,
                expert_id          TEXT NOT NULL,
                sop_text           TEXT,
                cumulative_return  REAL,
                sharpe             REAL,
                drawdown           REAL,
                uncertainty_score  REAL,
                extra_json         TEXT
            )
        """)
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_expert_id ON index_records(expert_id)"
        )
        self.con.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON index_records(timestamp)"
        )
        self.con.commit()
        logger.info("Opened SQLite database at %s", path)

    # ------------------------------------------------------------------
    # Add records
    # ------------------------------------------------------------------

    def add_record(
        self,
        embedding: np.ndarray,
        timestamp: str,
        expert_id: str,
        sop_text: str = "",
        cumulative_return: float = 0.0,
        sharpe: float = 0.0,
        drawdown: float = 0.0,
        uncertainty_score: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> int:
        """Add a single record to both FAISS and SQLite.

        Returns the FAISS / SQLite id assigned to the record.
        """
        if self.con is None:
            raise RuntimeError("SQLite database not opened – call load() or provide db_path.")

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        faiss_id = self._next_id
        self.index.add(embedding)
        self.con.execute(
            """
            INSERT INTO index_records
                (id, faiss_id, timestamp, expert_id, sop_text,
                 cumulative_return, sharpe, drawdown, uncertainty_score, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                faiss_id,
                faiss_id,
                timestamp,
                expert_id,
                sop_text,
                float(cumulative_return),
                float(sharpe),
                float(drawdown),
                float(uncertainty_score),
                json.dumps(extra) if extra else None,
            ),
        )
        self.con.commit()
        self._next_id += 1
        return faiss_id

    def add_records_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> list[int]:
        """Add multiple records in a single batch.

        ``metadata_list`` must have the same length as the first dimension
        of ``embeddings``.  Each dict should contain the same keys as
        :meth:`add_record`.
        """
        ids: list[int] = []
        for i, meta in enumerate(metadata_list):
            emb = embeddings[i] if embeddings.ndim == 2 else embeddings
            eid = self.add_record(
                embedding=emb,
                timestamp=meta.get("timestamp", ""),
                expert_id=meta.get("expert_id", ""),
                sop_text=meta.get("sop_text", ""),
                cumulative_return=meta.get("cumulative_return", 0.0),
                sharpe=meta.get("sharpe", 0.0),
                drawdown=meta.get("drawdown", 0.0),
                uncertainty_score=meta.get("uncertainty_score", 0.0),
                extra=meta.get("extra"),
            )
            ids.append(eid)
        return ids

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        expert_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-*k* metadata records closest to ``embedding``.

        Parameters
        ----------
        embedding :
            Query vector of shape ``(dim,)``.
        k :
            Number of nearest neighbours to retrieve.
        expert_id :
            If provided, only return records matching this expert.

        Returns
        -------
        List of dicts with SQLite columns plus a ``distance`` key.
        """
        if self.con is None:
            raise RuntimeError("SQLite database not opened.")
        if self.index.ntotal == 0:
            return []

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(embedding, k)
        results: list[dict[str, Any]] = []

        for dist, faiss_id in zip(distances[0], indices[0]):
            if faiss_id == -1:
                continue
            cursor = self.con.execute(
                "SELECT * FROM index_records WHERE faiss_id = ?", (int(faiss_id),)
            )
            row = cursor.fetchone()
            if row is None:
                continue
            cols = [desc[0] for desc in cursor.description]
            record = dict(zip(cols, row))
            record["distance"] = float(dist)
            if expert_id is not None and record["expert_id"] != expert_id:
                continue
            if record.get("extra_json"):
                record["extra"] = json.loads(record["extra_json"])
            else:
                record["extra"] = {}
            results.append(record)

        return results

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, index_path: str | Path, db_path: str | Path) -> None:
        """Persist the FAISS index and SQLite database to disk."""
        index_path = Path(index_path)
        db_path = Path(db_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))
        logger.info("FAISS index saved to %s (size %d)", index_path, self.index.ntotal)

        if self.con:
            self.con.commit()
            # If the DB was opened from a different path or in-memory, write it to disk
            if self._db_path is None or str(self._db_path) != str(db_path):
                backup = sqlite3.connect(str(db_path))
                with backup:
                    self.con.backup(backup)
                backup.close()
        logger.info("Metadata saved to %s", db_path)

    def load(self, index_path: str | Path, db_path: str | Path) -> None:
        """Load a previously saved index and database from disk."""
        self._load_index(Path(index_path))
        self._open_db(Path(db_path))
        self._next_id = self.index.ntotal

    def _load_index(self, path: Path) -> None:
        if path.exists():
            self.index = faiss.read_index(str(path))
            self.dim = self.index.d
            self._next_id = self.index.ntotal
            logger.info("FAISS index loaded from %s (size %d)", path, self.index.ntotal)
        else:
            logger.warning("FAISS index not found at %s – starting fresh", path)

    def __len__(self) -> int:
        return self.index.ntotal

    def close(self) -> None:
        if self.con:
            self.con.close()
            self.con = None
