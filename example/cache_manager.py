"""
Stage-level cache manager for the hydrodynamic-ANN pipeline.

* One SQLite DB (`output/run_cache.sqlite`) stores:
    key-hash  →  model.h5  +  x-validation CSV prefix
* Collision probability is negligible (SHA-256 of full recipe).
"""

import json, hashlib, sqlite3, datetime as dt, shutil, glob
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from shutil import SameFileError


class CacheManager:
    # ------------------------------------------------------------------ #
    def __init__(self, db_path: str = "output/run_cache.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS stages(
                   key_hash    TEXT PRIMARY KEY,
                   stage_name  TEXT,
                   parent_hash TEXT,
                   recipe_json TEXT,
                   model_path  TEXT,
                   prefix_path TEXT,
                   created_at  TEXT)"""
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hash(obj: Dict[str, Any]) -> str:
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def _row(self, h: str):
        cur = self.conn.execute("SELECT * FROM stages WHERE key_hash=?", (h,))
        return cur.fetchone()

    def _register(self, payload: Dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO stages
               (key_hash,stage_name,parent_hash,recipe_json,
                model_path,prefix_path,created_at)
               VALUES (:key_hash,:stage_name,:parent_hash,:recipe_json,
                       :model_path,:prefix_path,:created_at)""",
            payload,
        )
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def check(
        self,
        stage_name : str,
        recipe     : Dict[str, Any],
        model_path : str,          # absolute path, *without* ".h5"
        prefix_path: str,          # absolute prefix, *without* "_xvalid*.csv"
        parent_hash: Optional[str],
    ) -> Tuple[bool, Any]:
        """
        Returns
        -------
        hit : bool
            True  → artefacts copied, training should be skipped.
            False → caller must train and then call `finalize()`.
        info : if hit  → key_hash (str)
               if miss → (key_hash, finalize_callable)
        """
        recipe_full = dict(recipe, stage_name=stage_name, parent_hash=parent_hash)
        key_hash    = self._hash(recipe_full)

        # ------------------------------  CACHE HIT?  --------------------- #
        row = self._row(key_hash)

        # (a) Row exists but artefacts were deleted → purge & treat as miss
        if row:
            src_model = Path(row[4])                # …/model.h5 on disk
            if not src_model.exists():
                # Stale entry – remove it and fall through to MISS logic
                self.conn.execute("DELETE FROM stages WHERE key_hash=?", (key_hash,))
                self.conn.commit()
                row = None

        # (b) Proper cache hit – copy artefacts (if needed) ---------------
        if row:
            src_model = Path(row[4])
            dst_model = Path(f"{model_path}.h5")
            dst_model.parent.mkdir(parents=True, exist_ok=True)

            # Copy model unless src==dst
            try:
                if src_model.resolve() != dst_model.resolve():
                    shutil.copy2(src_model, dst_model)
            except SameFileError:
                # Identical file – nothing to do
                pass

            # Side‑car weights file
            src_wts = src_model.with_suffix(".weights.h5")
            if src_wts.exists():
                dst_wts = dst_model.with_suffix(".weights.h5")
                try:
                    if src_wts.resolve() != dst_wts.resolve():
                        shutil.copy2(src_wts, dst_wts)
                except SameFileError:
                    pass

            # Copy every x‑valid CSV
            for src_csv in glob.glob(f"{row[5]}_xvalid_*.csv"):
                dst_csv = Path(src_csv.replace(row[5], prefix_path))
                dst_csv.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if Path(src_csv).resolve() != dst_csv.resolve():
                        shutil.copy2(src_csv, dst_csv)
                except SameFileError:
                    pass

            return True, key_hash   # genuine HIT

        # ------------------------------  CACHE MISS  --------------------- #
        def finalize() -> None:
            """Register artefacts *after* successful training."""
            self._register(
                dict(
                    key_hash    = key_hash,
                    stage_name  = stage_name,
                    parent_hash = parent_hash,
                    recipe_json = json.dumps(recipe_full, indent=2, sort_keys=True),
                    model_path  = f"{model_path}.h5",
                    prefix_path = prefix_path,
                    created_at  = dt.datetime.utcnow().isoformat(timespec="seconds"),
                )
            )

        return False, (key_hash, finalize)