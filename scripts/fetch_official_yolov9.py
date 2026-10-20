"""Fetch only the configured official repository/release; validate before use."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/yolov9_official.json").read_text())
    repo = ROOT / "third_party/yolov9"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", config["repository"]], check=True)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", config["revision"]], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != config["revision"] or dirty:
        raise RuntimeError("Existing official checkout differs; refusing to overwrite")
    weight = ROOT / "data/checkpoints/yolov9-s-converted.pt"
    weight.parent.mkdir(parents=True, exist_ok=True)
    if weight.exists():
        if hashlib.sha256(weight.read_bytes()).hexdigest() != config["weights_sha256"]:
            raise RuntimeError("Existing checkpoint checksum differs; refusing to overwrite")
    else:
        temporary = weight.with_suffix(".download")
        try:
            with urllib.request.urlopen(config["weights_url"], timeout=120) as response, temporary.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != config["weights_sha256"]:
                raise RuntimeError("Downloaded checkpoint checksum mismatch")
            temporary.rename(weight)
        finally:
            temporary.unlink(missing_ok=True)
    print("Official YOLOv9 source and checkpoint verified")


if __name__ == "__main__":
    main()