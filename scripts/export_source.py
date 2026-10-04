"""Export the current publishable working tree, respecting .gitignore."""
import argparse
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/needle-gpu-source.tar.gz")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = (root / args.output).resolve()
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others",
                                     "--exclude-standard", "-z"], cwd=root).split(b"\0")
    files = sorted({name.decode() for name in paths if name and (root / name.decode()).is_file()
                    and (root / name.decode()).resolve() != destination})
    forbidden = {".git", ".venv", ".local_archive", "build", "__pycache__", ".pytest_cache", "dist"}
    for name in files:
        path = Path(name)
        if forbidden.intersection(path.parts) or path.suffix in {".so", ".o", ".pyc"}:
            raise SystemExit(f"Refusing to export generated/private file: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name in files:
            archive.add(root / name, arcname=f"needle-gpu/{name}", recursive=False)
    print(f"Exported {len(files)} files to {destination}")


if __name__ == "__main__":
    main()
