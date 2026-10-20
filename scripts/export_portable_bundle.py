"""Create a non-destructive migration archive; exclude environments and secrets.

Rewrite local absolute USD asset references ONLY in archived .usda copies.
Never modify live scenes or copy existing motion-validation approvals.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("configs", "docs", "src", "scripts", "tests", "isaac", "ros2_ws/src", "tableimage", "data")
ROOT_FILES = ("README.md", ".gitignore", ".dockerignore", "Dockerfile", "compose.yaml", "Makefile")
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".log", ".aux", ".fls", ".fdb_latexmk", ".toc", ".out", ".pem", ".key"}


def portable_usda(path, data):
    text = data.decode("utf-8")
    changed = []

    def reference(match):
        value = match.group(1)
        if not value.startswith("/"):
            return match.group(0)
        original = Path(value)
        try:
            original.relative_to(ROOT)
        except ValueError:
            raise RuntimeError(f"External local asset must be packaged manually: {value}") from None
        if not original.exists():
            raise FileNotFoundError(f"Missing scene dependency: {value}")
        relative = os.path.relpath(original, path.parent)
        changed.append({"from": value, "to": relative})
        return "@" + relative + "@"

    text = re.sub(r"@([^@\n]+)@", reference, text)
    return text.encode("utf-8"), changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "transfer" / f"ROBOT_VLA-{stamp}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()}
    for folder in FOLDERS:
        base = ROOT / folder
        if not base.exists():
            continue
        for directory, children, filenames in os.walk(base, followlinks=False):
            children[:] = sorted(n for n in children if n not in SKIP_DIRS and not (Path(directory) / n).is_symlink())
            for name in filenames:
                path = Path(directory) / name
                if path.is_symlink():
                    raise RuntimeError(f"Symlink requires manual review before export: {path}")
                if name.startswith(".env") or path.suffix in SKIP_SUFFIXES or name.endswith(".synctex.gz"):
                    continue
                paths.add(path)
    records, edits = [], {}
    # Exclusive open prevents overwriting a previous migration archive.
    with output.open("xb") as destination, tarfile.open(fileobj=destination, mode="w:gz") as archive:
        for path in sorted(paths):
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            if path.suffix == ".usda":
                data, changes = portable_usda(path, data)
                if changes:
                    edits[relative] = changes
            info = tarfile.TarInfo("ROBOT_VLA/" + relative)
            info.size, info.mode = len(data), 0o644
            archive.addfile(info, io.BytesIO(data))
            records.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"completed": True, "created_utc": stamp, "files": records,
                    "rewritten_usda_references": edits,
                    "excluded": ["virtual environments", "Git metadata", "CVAT database/volumes", "ROS build/install/log", "third_party checkout"],
                    "notes": ["No live files changed.", "Reinstall Isaac Sim on the destination; CPU Docker image is built separately.",
                              "Remote USD references still require network access.", "Motion must be freshly validated on destination."]}
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        info = tarfile.TarInfo("ROBOT_VLA/TRANSFER_MANIFEST.json")
        info.size, info.mode = len(data), 0o644
        archive.addfile(info, io.BytesIO(data))
    with output.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    print(json.dumps({"archive": str(output), "files": len(records), "sha256": digest,
                      "usda_files_rewritten_in_archive_only": len(edits)}, indent=2))


if __name__ == "__main__":
    main()