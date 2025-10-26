#!/usr/bin/env python
# ============================================================
#  purge_amended.py
#  ------------------------------------------------------------
#  Deletes ONLY the *_amended.csv files created by
#  amend_results.py so you can roll back instantly.
# ============================================================
import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger(__name__)


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Remove *_amended.csv files produced by amend_results.py")
    ap.add_argument("--config", required=True, help="Same amend_paths.yml used for amendment")
    return ap.parse_args()


def purge(root: Path, specs: list[dict]) -> None:
    removed = 0
    for spec in specs:
        # trial dirs
        pattern = spec["trial_dir_tpl"].format(trial="*")
        for trial_dir in root.glob(pattern):
            for f in trial_dir.glob("*_amended.csv"):
                f.unlink()
                LOG.debug("Deleted %s", f)
                removed += 1

        # master
        master = root / spec["master"]
        amended_master = master.with_name(f"{master.stem}_amended.csv")
        if amended_master.exists():
            amended_master.unlink()
            LOG.debug("Deleted %s", amended_master)
            removed += 1
    LOG.info("Total amended files removed: %d", removed)


def main() -> None:
    args = parse_cli()
    project_root = Path(__file__).resolve().parent
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    purge(project_root, cfg["approaches"])


if __name__ == "__main__":
    main()