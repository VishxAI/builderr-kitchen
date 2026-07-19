"""Local VLM for fine-grained frame inspection (Qwen3-VL-2B-Instruct).

Lazy-loads once, answers targeted questions about single frames or short
frame sequences. Runs on GPU (bf16) when available, CPU otherwise.
Zero API cost — this is the "fine pass" engine.

CLI smoke test:
    python -m pipeline.vlm videos/check_600.jpg "How many people are visible?"
"""

import argparse
import sys

import cv2
import torch

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is not None:
        return
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device
    )
    _processor = AutoProcessor.from_pretrained(MODEL_ID)


def ask(images, question: str, max_new_tokens: int = 128) -> str:
    """Ask a question about one or more BGR frames (numpy arrays) or file paths.

    Returns the model's text answer. Multiple images are presented in order,
    which lets the caller ask about a short temporal sequence.
    """
    _load()
    if not isinstance(images, (list, tuple)):
        images = [images]

    content = []
    for img in images:
        if isinstance(img, str):
            content.append({"type": "image", "image": img})
        else:
            from PIL import Image

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            content.append({"type": "image", "image": Image.fromarray(rgb)})
    content.append({"type": "text", "text": question})

    messages = [{"role": "user", "content": content}]
    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.inference_mode():
        out = _model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return _processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local VLM about an image")
    parser.add_argument("image")
    parser.add_argument("question")
    args = parser.parse_args()
    print(ask(args.image, args.question))


if __name__ == "__main__":
    main()
