"""Verify MediaRegistry compilation, local-chain behavior, and API integration."""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def _check_imports() -> None:
    packages = {
        "web3": "web3",
        "solcx": "py-solc-x",
        "eth_tester": "eth-tester",
        "eth": "py-evm",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "multipart": "python-multipart",
        "requests": "requests",
        "PIL": "Pillow",
        "pandas": "pandas",
        "torch": "torch",
        "torchvision": "torchvision",
    }
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name or error.name.startswith(f"{module_name}."):
                raise ModuleNotFoundError(
                    f"Required package is missing: {package_name} (import {module_name})"
                ) from error
            raise


def _pick_images(root: Path) -> tuple[Path, Path]:
    import pandas as pd

    manifest_path = root / "data" / "faces" / "test_2k.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if not {"path", "label"}.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} must contain path,label columns")
    selected: dict[int, Path] = {}
    for label in (0, 1):
        candidates = manifest.loc[manifest["label"] == label]
        if candidates.empty:
            raise ValueError(f"No test images found for label {label}")
        candidate = Path(candidates.sample(n=1, random_state=42).iloc[0]["path"])
        if not candidate.is_file():
            raise FileNotFoundError(f"Test image not found: {candidate}")
        selected[label] = candidate
    return selected[0], selected[1]


def _tamper_as_png(source_bytes: bytes, path: Path) -> bytes:
    from PIL import Image

    with Image.open(io.BytesIO(source_bytes)) as source:
        image = source.convert("RGB")
    pixels = image.load()
    red, green, blue = pixels[0, 0]
    pixels[0, 0] = (255 - red, green, blue)
    image.save(path, format="PNG", optimize=False)
    tampered_bytes = path.read_bytes()
    if hashlib.sha256(tampered_bytes).digest() == hashlib.sha256(source_bytes).digest():
        raise AssertionError("Tampered PNG unexpectedly has the original raw-byte hash")
    return tampered_bytes


def _start_server(root: Path) -> tuple[subprocess.Popen[bytes], str, Any]:
    import requests

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    base_url = f"http://127.0.0.1:{port}"
    log = tempfile.TemporaryFile(mode="w+b")
    process = subprocess.Popen(
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
        cwd=root,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 120
    while True:
        if process.poll() is not None:
            log.seek(0)
            details = log.read().decode("utf-8", errors="replace")
            log.close()
            raise RuntimeError(f"Uvicorn exited with status {process.returncode}:\n{details}")
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.ok:
                print("API /health:", response.json())
                return process, base_url, log
        except requests.RequestException:
            pass
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait(timeout=15)
            log.seek(0)
            details = log.read().decode("utf-8", errors="replace")
            log.close()
            raise TimeoutError(f"Uvicorn readiness timed out:\n{details}")
        time.sleep(0.5)


def _stop_server(process: subprocess.Popen[bytes], log: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    log.close()
    print("API server stopped.")


def main() -> int:
    _check_imports()
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import requests
    from eth_tester.exceptions import TransactionFailed
    from PIL import Image
    from web3.exceptions import ContractLogicError

    from src.blockchain import ProvenanceClient

    fake_path, real_path = _pick_images(root)
    direct_client = ProvenanceClient()
    direct_client.deploy()
    results: list[dict[str, str]] = []
    image_bytes: dict[str, bytes] = {
        "fake": fake_path.read_bytes(),
        "real": real_path.read_bytes(),
    }
    hashes = {
        label: hashlib.sha256(content).hexdigest()
        for label, content in image_bytes.items()
    }

    for label, is_fake in (("fake", True), ("real", False)):
        direct_client.register(hashes[label], is_fake, 0.75 if is_fake else 0.25)
        record = direct_client.verify(hashes[label])
        if not record["registered"] or record["is_fake"] is not is_fake:
            raise AssertionError(f"Direct chain verification failed for {label}")
        results.append(
            {
                "test": f"register {label}",
                "true label": label,
                "result": "REGISTERED",
                "details": f"p_fake={record['p_fake']:.4f}",
            }
        )

    copied_record = direct_client.verify(hashlib.sha256(bytes(image_bytes["real"])).hexdigest())
    if not copied_record["registered"]:
        raise AssertionError("Exact byte copy should verify as registered")
    results.append(
        {"test": "exact copy", "true label": "real", "result": "REGISTERED", "details": "same raw bytes"}
    )

    with tempfile.TemporaryDirectory(prefix="stage_b_tamper_") as temporary_directory:
        tampered_path = Path(temporary_directory) / "tampered.png"
        tampered_bytes = _tamper_as_png(image_bytes["real"], tampered_path)
        tampered_record = direct_client.verify(hashlib.sha256(tampered_bytes).hexdigest())
        if tampered_record["registered"]:
            raise AssertionError("One-pixel modified file unexpectedly matched the registered hash")
        results.append(
            {"test": "one-pixel tamper", "true label": "real", "result": "NOT FOUND", "details": "hash changed"}
        )

    try:
        direct_client.register(hashes["real"], False, 0.25)
    except (TransactionFailed, ContractLogicError, ValueError):
        results.append(
            {"test": "duplicate registration", "true label": "real", "result": "REVERTED", "details": "already registered"}
        )
    else:
        raise AssertionError("Duplicate registration did not revert")

    invalid_hash_rejected = False
    try:
        direct_client.verify("not-a-sha256")
    except ValueError:
        invalid_hash_rejected = True
    if not invalid_hash_rejected:
        raise AssertionError("Invalid SHA-256 input was not rejected")

    api_image = image_bytes["fake"]
    api_filename = fake_path.name
    api_content_type = mimetypes.guess_type(api_filename)[0] or "image/jpeg"
    server, base_url, server_log = _start_server(root)
    try:
        analyze_response = requests.post(
            f"{base_url}/analyze",
            data={"register": "true"},
            files={"image": (api_filename, api_image, api_content_type)},
            timeout=180,
        )
        if analyze_response.status_code != 200:
            raise RuntimeError(
                f"POST /analyze register=true failed: "
                f"{analyze_response.status_code} {analyze_response.text}"
            )
        analyzed = analyze_response.json()
        if not analyzed.get("tx_hash"):
            raise AssertionError("/analyze did not return a registration transaction hash")
        results.append(
            {
                "test": "API analyze + register",
                "true label": "fake",
                "result": analyzed["verdict"],
                "details": f"p_fake={float(analyzed['p_fake']):.4f}",
            }
        )

        verify_response = requests.post(
            f"{base_url}/verify",
            files={"image": (api_filename, api_image, api_content_type)},
            timeout=30,
        )
        if verify_response.status_code != 200:
            raise RuntimeError(f"POST /verify failed: {verify_response.status_code} {verify_response.text}")
        verified = verify_response.json()
        if not verified["registered"] or verified["sha256"] != analyzed["sha256"]:
            raise AssertionError("API verification of exact uploaded bytes did not find the record")
        results.append(
            {
                "test": "API verify original/copy",
                "true label": "fake",
                "result": "REGISTERED",
                "details": verified["verdict"],
            }
        )

        with tempfile.TemporaryDirectory(prefix="stage_b_api_tamper_") as temporary_directory:
            tampered_path = Path(temporary_directory) / "tampered.png"
            api_tampered = _tamper_as_png(api_image, tampered_path)
            tampered_response = requests.post(
                f"{base_url}/verify",
                files={"image": ("tampered.png", api_tampered, "image/png")},
                timeout=30,
            )
        if tampered_response.status_code != 200:
            raise RuntimeError(
                f"POST /verify tampered file failed: "
                f"{tampered_response.status_code} {tampered_response.text}"
            )
        api_tampered_result = tampered_response.json()
        if api_tampered_result["registered"] or "not registered, or the file was modified" not in api_tampered_result["message"]:
            raise AssertionError("API tamper verification returned unexpected result/message")
        results.append(
            {
                "test": "API verify one-pixel tamper",
                "true label": "fake",
                "result": "NOT FOUND",
                "details": "hash changed",
            }
        )
    finally:
        _stop_server(server, server_log)

    print("\nStage B provenance results:")
    import pandas as pd

    print(pd.DataFrame(results).to_string(index=False))

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "requirements.txt").write_text(freeze.stdout, encoding="utf-8", newline="")
    print(f"\nUpdated requirements.txt ({len(freeze.stdout.splitlines())} packages).")
    print("Verified package imports, Solidity compile/deploy, direct contract cases, and API cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
