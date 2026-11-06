"""Run a local API smoke test and save representative Grad-CAM examples."""

from __future__ import annotations

import base64
import io
import mimetypes
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "faces" / "test_2k.csv"
EXAMPLES_DIR = PROJECT_ROOT / "docs" / "examples"
MAX_SERVER_START_SECONDS = 90


def _available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sample_rows() -> pd.DataFrame:
    manifest = pd.read_csv(CSV_PATH)
    if not {"path", "label"}.issubset(manifest.columns):
        raise ValueError(f"{CSV_PATH} must contain path,label columns")
    if not manifest["label"].isin([0, 1]).all():
        raise ValueError(f"{CSV_PATH} labels must be 0 (fake) or 1 (real)")
    selected = []
    for label in (1, 0):
        class_rows = manifest.loc[manifest["label"] == label]
        if len(class_rows) < 3:
            raise ValueError(f"Need at least three examples with label {label}")
        selected.append(class_rows.sample(n=3, random_state=42))
    return pd.concat(selected, ignore_index=True)


def main() -> int:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Test manifest not found: {CSV_PATH}")

    port = _available_port()
    base_url = f"http://127.0.0.1:{port}"
    server_log = tempfile.TemporaryFile(mode="w+b")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + MAX_SERVER_START_SECONDS
        while True:
            if server.poll() is not None:
                server_log.seek(0)
                log = server_log.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Uvicorn exited with status {server.returncode}:\n{log}")
            try:
                response = requests.get(f"{base_url}/health", timeout=2)
                if response.ok:
                    health = response.json()
                    break
            except requests.RequestException:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Uvicorn did not become ready within 90 seconds")
            time.sleep(0.5)

        print("GET /health:", health)
        if health.get("device") != "cpu":
            raise AssertionError(f"Expected CPU inference, got {health.get('device')!r}")

        rows = _sample_rows()
        results: list[dict[str, str | float]] = []
        saved_labels: set[int] = set()
        for row in rows.itertuples(index=False):
            image_path = Path(row.path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Test image not found: {image_path}")
            content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            with image_path.open("rb") as image_file:
                response = requests.post(
                    f"{base_url}/analyze",
                    files={"image": (image_path.name, image_file, content_type)},
                    timeout=90,
                )
            if response.status_code != 200:
                raise RuntimeError(
                    f"/analyze failed for {image_path.name}: "
                    f"{response.status_code} {response.text}"
                )
            payload = response.json()
            results.append(
                {
                    "file": image_path.name,
                    "true label": "real" if int(row.label) == 1 else "fake",
                    "verdict": payload["verdict"],
                    "p_fake": round(float(payload["p_fake"]), 4),
                }
            )

            label = int(row.label)
            if label not in saved_labels:
                png_bytes = base64.b64decode(payload["heatmap_png_base64"], validate=True)
                with Image.open(io.BytesIO(png_bytes)) as heatmap:
                    heatmap = heatmap.convert("RGB")
                    heatmap.thumbnail((256, 256))
                    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
                    name = "real" if label == 1 else "fake"
                    heatmap.save(EXAMPLES_DIR / f"heatmap_{name}.png", format="PNG", optimize=True)
                saved_labels.add(label)

        print("\nPOST /analyze results (seed 42):")
        print(pd.DataFrame(results).to_string(index=False))
        print("\nSaved Grad-CAM examples:")
        for name in ("real", "fake"):
            output_path = EXAMPLES_DIR / f"heatmap_{name}.png"
            if not output_path.is_file():
                raise RuntimeError(f"Expected heatmap was not saved: {output_path}")
            print(f"{output_path.relative_to(PROJECT_ROOT)} ({output_path.stat().st_size:,} bytes)")

    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log.close()
        print("Uvicorn stopped.")

    frozen_requirements = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (PROJECT_ROOT / "requirements.txt").write_text(
        frozen_requirements.stdout,
        encoding="utf-8",
        newline="",
    )
    print(f"Updated requirements.txt ({len(frozen_requirements.stdout.splitlines())} packages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
