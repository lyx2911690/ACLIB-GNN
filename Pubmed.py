"""Compatibility entry point; original implementation is in legacy/."""
import sys
from train import main

if __name__ == "__main__":
    main(["--dataset", "Pubmed", *sys.argv[1:]])
