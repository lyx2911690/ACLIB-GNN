"""Dataset shortcut; run from any working directory."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train import main

if __name__ == "__main__":
    main(["--dataset", "Citeseer", *sys.argv[1:]])
