"""FastAPI service for face authenticity inference and Grad-CAM."""

from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from web3.exceptions import ContractLogicError
from eth_tester.exceptions import TransactionFailed

from src.blockchain import get_provenance_client, sha256_to_bytes32
from src.gradcam import gradcam_overlay
from src.predict import load_model, predict_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

app = FastAPI(title="Deepfake Face Detector", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    configured = os.environ.get("MODEL_WEIGHTS", "models/effnet_b0_smoke.pth")
    return {
        "status": "ok",
        "model_weights": configured,
        "device": "cpu",
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    register: bool = Form(default=False),
) -> dict[str, str | float]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload must be a JPG or PNG image.")
    filename = Path(image.filename or "upload").name
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Upload must have a .jpg, .jpeg, or .png filename.")

    raw_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.verify()
            image_format = source.format
        if image_format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail="File content must be a valid JPG or PNG image.")
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.load()
            original = source.convert("RGB")
    except HTTPException:
        raise
    except UnidentifiedImageError as error:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a readable JPG or PNG image.",
        ) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {error}") from error

    try:
        model = load_model()
        result = predict_image(raw_bytes, model=model)
        overlay = gradcam_overlay(original, model, target_class=0)
        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG", optimize=True)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {error}") from error

    response: dict[str, str | float] = {
        "filename": filename,
        **result,
        "heatmap_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }
    if register:
        try:
            response["tx_hash"] = get_provenance_client().register(
                result["sha256"],
                is_fake=result["verdict"] == "FAKE",
                p_fake=result["p_fake"],
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        except (ContractLogicError, TransactionFailed) as error:
            raise HTTPException(status_code=409, detail="This file hash is already registered.") from error
        except (OSError, RuntimeError) as error:
            raise HTTPException(status_code=503, detail=f"Blockchain registration failed: {error}") from error
    return response


@app.post("/verify")
async def verify(image: UploadFile = File(...)) -> dict[str, str | float | bool | None]:
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload must be a JPG or PNG image.")
    filename = Path(image.filename or "upload").name
    if Path(filename).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=400, detail="Upload must have a .jpg, .jpeg, or .png filename.")
    raw_bytes = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.verify()
            if source.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=400, detail="File content must be a valid JPG or PNG image.")
    except HTTPException:
        raise
    except UnidentifiedImageError as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable JPG or PNG image.") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"Unreadable image: {error}") from error

    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    try:
        sha256_to_bytes32(sha256_hex)
        record = get_provenance_client().verify(sha256_hex)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=f"Blockchain verification failed: {error}") from error
    if not record["registered"]:
        return {
            "registered": False,
            "verdict": None,
            "p_fake": None,
            "registrar": None,
            "timestamp": None,
            "sha256": sha256_hex,
            "message": "This file is not registered, or the file was modified.",
        }
    return {
        "registered": True,
        "verdict": "FAKE" if record["is_fake"] else "REAL",
        "p_fake": record["p_fake"],
        "registrar": record["registrar"],
        "timestamp": record["timestamp"],
        "sha256": sha256_hex,
        "message": "This file hash is registered on-chain.",
    }
