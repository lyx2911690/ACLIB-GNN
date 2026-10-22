"""Build a source-only upload archive without local data or dependencies."""
from pathlib import Path
import hashlib
import json
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    names = ["README.md", "README.zh-CN.md", "requirements.txt", "environment.yml",
             ".gitignore", "train.py", "evaluate.py", "Cora.py", "Citeseer.py",
             "Pubmed.py", "CS.py"]
    if (root / "LICENSE").is_file():
        names.append("LICENSE")
    files = [root / name for name in names]
    for name in ["aclib", "configs", "docs", "tests", "legacy", ".github", "scripts"]:
        files.extend(p for p in (root / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in {".pyc", ".pyo"})
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    archive = root / "ACLIB-GNN-GitHub-source.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as package:
        for file in files:
            package.write(file, "ACLIB-GNN/" + file.relative_to(root).as_posix())
        package.writestr("ACLIB-GNN/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with ZipFile(archive) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(files) + 1
    print(f"Created {archive.name}: {len(files)} source files plus SHA-256 manifest")


if __name__ == "__main__":
    main()
