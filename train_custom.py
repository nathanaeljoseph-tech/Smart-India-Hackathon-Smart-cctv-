"""
train_custom.py - Auto Download + Train YOLOv11n for Border Surveillance
=========================================================================
Smart India Hackathon 2026 | Problem: SIH26187
Module: DEV 1 - Automated Training Pipeline

HOW TO RUN:
  python train_custom.py                    # Full auto: download + train
  python train_custom.py --epochs 30        # Fewer epochs (faster, less accurate)
  python train_custom.py --skip-download    # Re-train on already downloaded data
  python train_custom.py --skip-train       # Only download, no training

What this script does:
  1. Tries multiple border surveillance datasets from Roboflow Universe
     in order of relevance to SIH26187
  2. Patches data.yaml paths to absolute (Windows path safety)
  3. Trains yolo11n.pt with CPU-optimised hyperparameters
  4. Validates and prints mAP50 / Precision / Recall
  5. Copies best.pt to project root as border_best.pt

Datasets (tried in this order):
  1. college-k0pgy   / border-surveillance-7nfat        (drones, rifles, intruders)
  2. capstone-ozfof  / border-surveillance-7nfat-otktw  (alternate version)
  3. border-security / bordersecurity-jpjpx             (person + vehicle at border)

Author : DEV 1
Date   : 2026-09-05
"""

import os
import sys
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train_custom.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("train_custom")

# ---------------------------------------------------------------------------
# CONFIG ΓÇö edit these if you want to change targets
# ---------------------------------------------------------------------------
API_KEY = "aIP4EmhkU3kDXxgii9ZN"

# (workspace, project_id, version) ΓÇö tried in order until one succeeds
DATASET_CANDIDATES = [
    ("college-k0pgy",   "border-surveillance-7nfat",       1),
    ("capstone-ozfof",  "border-surveillance-7nfat-otktw",  1),
    ("border-security", "bordersecurity-jpjpx",             1),
]

try:
    import torch
    DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"
except Exception:
    DEFAULT_DEVICE = "cpu"

# ---------------------------------------------------------------------------
# Training Hyperparameters (optimized for SIH hackathon speed + accuracy)
# ---------------------------------------------------------------------------
TRAIN_CFG = {
    "model"         : "yolo11n.pt",   # Pretrained COCO nano weights
    "epochs"        : 50,             # Fine-tuning epochs
    "imgsz"         : 640,
    "batch"         : 8,              # Safe batch size
    "device"        : DEFAULT_DEVICE,
    "workers"       : 2,
    "patience"      : 15,             # Early stopping
    "lr0"           : 0.01,
    "lrf"           : 0.01,
    "momentum"      : 0.937,
    "weight_decay"  : 0.0005,
    "warmup_epochs" : 3,
    "cos_lr"        : True,
    "cache"         : False,
    "exist_ok"      : True,
    "plots"         : True,
    "save"          : True,
    "save_period"   : 10,
    "verbose"       : True,
}

DATASET_DIR = Path("datasets")
OUTPUT_DIR  = Path("border_net_training")


# ---------------------------------------------------------------------------
# Dataset Download
# ---------------------------------------------------------------------------
def download_dataset():
    """
    Try each Roboflow dataset candidate until one succeeds.
    Returns the Path to the dataset folder (which contains data.yaml),
    or None if all attempts failed.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        log.error("roboflow not installed. Run:  pip install roboflow")
        return None

    rf = Roboflow(api_key=API_KEY)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for workspace, project_id, version_num in DATASET_CANDIDATES:
        log.info("-" * 55)
        log.info(f"Trying: {workspace}/{project_id}  version {version_num}")
        log.info("-" * 55)
        try:
            project  = rf.workspace(workspace).project(project_id)
            dataset  = project.version(version_num).download(
                "yolov8",                                # yolov8 format = works with yolo11
                location=str(DATASET_DIR / project_id),
                overwrite=True,
            )
            base = Path(dataset.location)

            # Verify data.yaml exists
            yaml_p = base / "data.yaml"
            if yaml_p.exists():
                log.info(f"OK ΓÇö dataset at: {base}")
                _log_yaml(yaml_p)
                return base

            # Search recursively in case roboflow nested the files
            found = list(base.rglob("data.yaml"))
            if found:
                log.info(f"OK ΓÇö data.yaml found at: {found[0]}")
                _log_yaml(found[0])
                return found[0].parent

            log.warning("Download succeeded but data.yaml not found. Trying next...")

        except Exception as exc:
            log.warning(f"FAILED ({workspace}/{project_id}): {exc}")

    return None


def _log_yaml(yaml_path):
    """Print dataset class info from data.yaml."""
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        log.info(f"  Classes ({d.get('nc','?')}): {d.get('names', '?')}")
        log.info(f"  Train : {d.get('train', 'N/A')}")
        log.info(f"  Val   : {d.get('val',   'N/A')}")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path Patching
# ---------------------------------------------------------------------------
def patch_yaml(yaml_path: Path) -> Path:
    """
    Make all paths in data.yaml absolute.
    Prevents training failures when Python CWD differs from dataset folder.
    """
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        root    = yaml_path.parent
        changed = False
        for key in ("train", "val", "test"):
            val = data.get(key)
            if val and not Path(val).is_absolute():
                abs_p = (root / val).resolve()
                if abs_p.exists():
                    data[key] = str(abs_p)
                    changed   = True
                    log.info(f"  Patched [{key}] -> {abs_p}")

        if changed:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
            log.info("data.yaml saved with absolute paths.")

    except Exception as exc:
        log.warning(f"Could not patch data.yaml: {exc}")

    return yaml_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(data_yaml: Path):
    """Train YOLOv11n and return path to best.pt or None."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed. Run:  pip install ultralytics")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M')}"

    log.info("=" * 55)
    log.info("  STARTING YOLO11n TRAINING  (CPU mode)")
    log.info("=" * 55)
    log.info(f"  data.yaml : {data_yaml}")
    log.info(f"  epochs    : {TRAIN_CFG['epochs']}")
    log.info(f"  batch     : {TRAIN_CFG['batch']}")
    log.info(f"  run name  : {run_name}")
    log.info("  NOTE: CPU training is slow. ~5-30 min/epoch.")
    log.info("  Leave this window open and do NOT press Ctrl+C.")
    log.info("  Early stopping will kick in after 15 idle epochs.")
    log.info("=" * 55)

    model = YOLO(TRAIN_CFG["model"])

    model.train(
        data          = str(data_yaml),
        epochs        = TRAIN_CFG["epochs"],
        imgsz         = TRAIN_CFG["imgsz"],
        batch         = TRAIN_CFG["batch"],
        device        = TRAIN_CFG["device"],
        workers       = TRAIN_CFG["workers"],
        patience      = TRAIN_CFG["patience"],
        lr0           = TRAIN_CFG["lr0"],
        lrf           = TRAIN_CFG["lrf"],
        momentum      = TRAIN_CFG["momentum"],
        weight_decay  = TRAIN_CFG["weight_decay"],
        warmup_epochs = TRAIN_CFG["warmup_epochs"],
        cos_lr        = TRAIN_CFG["cos_lr"],
        cache         = TRAIN_CFG["cache"],
        exist_ok      = TRAIN_CFG["exist_ok"],
        plots         = TRAIN_CFG["plots"],
        save          = TRAIN_CFG["save"],
        save_period   = TRAIN_CFG["save_period"],
        verbose       = TRAIN_CFG["verbose"],
        project       = str(OUTPUT_DIR),
        name          = run_name,
        # Augmentation tuned for fixed surveillance cameras
        hsv_h         = 0.015,
        hsv_s         = 0.7,
        hsv_v         = 0.4,
        degrees       = 5.0,
        translate     = 0.1,
        scale         = 0.5,
        flipud        = 0.0,    # gravity is always down in CCTV footage
        fliplr        = 0.5,
        mosaic        = 1.0,
        mixup         = 0.0,
    )

    # Locate best.pt
    best = OUTPUT_DIR / run_name / "weights" / "best.pt"
    if best.exists():
        return best
    for p in OUTPUT_DIR.rglob("best.pt"):
        return p
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(best_pt: Path, data_yaml: Path):
    """Run val loop and print key metrics."""
    log.info("Running validation on best.pt ...")
    try:
        from ultralytics import YOLO
        m       = YOLO(str(best_pt))
        metrics = m.val(data=str(data_yaml), device=TRAIN_CFG["device"])
        log.info("--- VALIDATION RESULTS ---")
        log.info(f"  mAP50    : {metrics.box.map50:.4f}")
        log.info(f"  mAP50-95 : {metrics.box.map:.4f}")
        log.info(f"  Precision: {metrics.box.p[0]:.4f}")
        log.info(f"  Recall   : {metrics.box.r[0]:.4f}")
        log.info("--------------------------")
    except Exception as exc:
        log.warning(f"Validation error (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Auto-download Roboflow dataset and train YOLOv11n for SIH26187"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing datasets/ folder")
    parser.add_argument("--data-yaml", type=str, default=None,
                        help="Path to data.yaml (paired with --skip-download)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count (default: 50)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Download only, no training")
    args = parser.parse_args()

    if args.epochs:
        TRAIN_CFG["epochs"] = args.epochs

    log.info("=" * 55)
    log.info("  AMST Border-Net | YOLOv11n Training Pipeline")
    log.info("  Smart India Hackathon 2026 | SIH26187 | DEV 1")
    log.info("=" * 55)

    # ---- Download ----
    if args.skip_download:
        if args.data_yaml:
            data_yaml = Path(args.data_yaml)
        else:
            found = list(DATASET_DIR.rglob("data.yaml"))
            if not found:
                log.error(
                    "No data.yaml in datasets/. "
                    "Remove --skip-download to download fresh."
                )
                sys.exit(1)
            data_yaml = found[0]
            log.info(f"Using existing: {data_yaml}")
    else:
        ds_path = download_dataset()
        if ds_path is None:
            log.error(
                "All dataset candidates failed. Options:\n"
                "  1. Check your API key is active at app.roboflow.com\n"
                "  2. Visit universe.roboflow.com, find a border dataset,\n"
                "     add (workspace, project, version) to DATASET_CANDIDATES\n"
                "  3. Run with --skip-download if you have a local dataset"
            )
            sys.exit(1)
        data_yaml = ds_path / "data.yaml"

    data_yaml = patch_yaml(data_yaml)

    if args.skip_train:
        log.info(f"Dataset ready. Skipping training.")
        log.info(f"data.yaml : {data_yaml}")
        return

    # ---- Train ----
    best_pt = train(data_yaml)
    if best_pt is None:
        log.error("Training failed. Check train_custom.log for details.")
        sys.exit(1)

    # ---- Validate ----
    validate(best_pt, data_yaml)

    # ---- Copy to project root ----
    dest = Path("border_best.pt")
    try:
        shutil.copy2(best_pt, dest)
        log.info(f"Best weights copied to: {dest.resolve()}")
    except Exception as exc:
        log.warning(f"Could not copy weights: {exc}")

    log.info("=" * 55)
    log.info("  TRAINING COMPLETE!")
    log.info(f"  Best weights  : {best_pt.resolve()}")
    log.info(f"  Quick copy    : border_best.pt")
    log.info("  To run with your model:")
    log.info("    python main.py --model border_best.pt")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
