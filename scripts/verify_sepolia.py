"""Register and verify a real image hash on the configured Sepolia registry."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from PIL import Image

from src.blockchain import SEPOLIA_CHAIN_ID, ProvenanceClient

MANIFEST_PATH = PROJECT_ROOT / "data" / "faces" / "test_2k.csv"
ETHERSCAN_TRANSACTION_URL = "https://sepolia.etherscan.io/tx/"


def _real_image_path() -> Path:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("data/faces/test_2k.csv was not found.")
    with MANIFEST_PATH.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if not {"path", "label"}.issubset(reader.fieldnames or ()):
            raise ValueError("The test manifest must contain path and label columns.")
        for row in reader:
            if row["label"] == "1":
                image_path = Path(row["path"])
                if image_path.is_file():
                    return image_path
    raise FileNotFoundError("No accessible real-labeled image was found in the test manifest.")


def _one_pixel_modified_png(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
    red, green, blue = image.getpixel((0, 0))
    image.putpixel((0, 0), (255 - red, green, blue))
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _verify() -> None:
    if not os.environ.get("RPC_URL") or not os.environ.get("PRIVATE_KEY"):
        raise ValueError("RPC_URL and PRIVATE_KEY must be configured in .env.")
    if not os.environ.get("CONTRACT_ADDRESS"):
        raise ValueError("CONTRACT_ADDRESS must be configured in .env.")

    client = ProvenanceClient()
    if client.web3.eth.chain_id != SEPOLIA_CHAIN_ID:
        raise RuntimeError("The configured RPC endpoint is not connected to Sepolia.")
    if client.contract is None:
        raise RuntimeError("The Sepolia MediaRegistry contract is not configured.")

    image_bytes = _real_image_path().read_bytes()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    transaction_hash = client.register(image_hash, is_fake=False, p_fake=0.0)
    print(f"Registration transaction: {transaction_hash}")
    print(f"Transaction explorer: {ETHERSCAN_TRANSACTION_URL}{transaction_hash}")

    registered = client.verify(image_hash)["registered"]
    if not registered:
        raise AssertionError("The registered image hash was not found on-chain.")

    modified_hash = hashlib.sha256(_one_pixel_modified_png(image_bytes)).hexdigest()
    tampered_registered = client.verify(modified_hash)["registered"]
    if tampered_registered:
        raise AssertionError("The one-pixel-modified image unexpectedly matched the registered hash.")

    print("\nVerification results:")
    print("| Image | Result |")
    print("|---|---|")
    print("| registered | REGISTERED |")
    print("| tampered | NOT FOUND |")


def main() -> int:
    try:
        _verify()
    except Exception as error:
        print(f"Sepolia verification failed ({type(error).__name__}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
