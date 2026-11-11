"""
Standardized Experiment Logging and Artifact Tracking Utilities (Phase 3.1)
Project: An Uncertainty-Aware, Budget-Constrained Multi-View Framework for Cost-Efficient Academic Integrity Detection
"""

from pathlib import Path
import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research_upgrade.src.config import (
    PHASE3_ARTIFACTS_DIR,
    PHASE3_RUNS_DIR,
    PHASE3_LOGS_DIR,
    RANDOM_SEED
)


def setup_logger(
    name: str = "academic_integrity",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures a standardized logger with console and file handlers.
    Ensures idempotency by not duplicating handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (if log_file specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def create_experiment_run(
    experiment_name: str,
    phase_tag: str = "phase_3_1",
    seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """
    Initializes a new timestamped experimental run directory under artifacts/phase_3/runs/.
    Creates run directory, metadata json, and dedicated run log.
    
    Naming Convention:
      Run Directory:  {YYYYMMDD_HHMMSS}_{phase_tag}_{experiment_name}
      Config File:    run_config.json
      Log File:       run_execution.log
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = experiment_name.lower().replace(" ", "_").replace("-", "_")
    run_id = f"{timestamp}_{phase_tag}_{clean_name}"
    run_dir = PHASE3_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run_execution.log"
    run_logger = setup_logger(name=run_id, log_file=log_path)

    run_metadata = {
        "run_id": run_id,
        "experiment_name": experiment_name,
        "phase_tag": phase_tag,
        "created_at": datetime.now().isoformat(),
        "random_seed": seed,
        "run_directory": str(run_dir),
        "log_path": str(log_path),
        "status": "INITIALIZED"
    }

    # Save initial metadata
    metadata_path = run_dir / "run_config.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    run_logger.info(f"Initialized experiment run: {run_id}")
    run_logger.info(f"Run directory created at: {run_dir}")
    run_logger.info(f"Deterministic seed set to: {seed}")

    return {
        "metadata": run_metadata,
        "logger": run_logger,
        "run_dir": run_dir,
        "log_path": log_path,
        "metadata_path": metadata_path
    }


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """Saves dictionary to JSON with utf-8 encoding and parent directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_json(path: Path) -> Dict[str, Any]:
    """Loads dictionary from JSON with utf-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

