"""Selective OCR for labels, receipts, and order-number questions (EasyOCR, local).

Used sparingly per the challenge guidance — only on frames the router flags
as text-bearing. First call downloads the detector/recognizer weights once.

CLI smoke test:
    python -m pipeline.ocr videos/check_600.jpg
"""

import argparse

import cv2
import torch

_reader = None


def _load():
    global _reader
    if _reader is None:
        import easyocr

        _reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
    return _reader


def read_text(image, min_confidence: float = 0.3, upscale: float = 2.0) -> list[dict]:
    """OCR a BGR frame (numpy array) or image path.

    Returns [{"text": str, "confidence": float, "box": [[x,y]x4]}] sorted by confidence.
    CCTV text is small, so the frame is upscaled before recognition.
    """
    reader = _load()
    if isinstance(image, str):
        image = cv2.imread(image)
    if upscale and upscale != 1.0:
        image = cv2.resize(image, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    # CCTV text can sit at any angle (vertical stove markings, tilted labels)
    results = reader.readtext(image, rotation_info=[90, 180, 270])
    out = [
        {"text": text, "confidence": round(float(conf), 3),
         "box": [[round(x / upscale, 1), round(y / upscale, 1)] for x, y in box]}
        for box, text, conf in results
        if conf >= min_confidence
    ]
    return sorted(out, key=lambda r: -r["confidence"])


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR an image")
    parser.add_argument("image")
    args = parser.parse_args()
    for r in read_text(args.image):
        print(f"{r['confidence']:.2f}  {r['text']}")


if __name__ == "__main__":
    main()
