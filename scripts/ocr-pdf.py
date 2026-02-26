#!/usr/bin/env python3
"""ocr-pdf.py — OCR extraction from PDF or image files via Tesseract.

Usage: python3 ocr-pdf.py --file <path> [--lang eng] [--min-text 100]
Output: JSON to stdout: {"text":"...", "pages":N, "method":"pypdf|tesseract", "chars":N}
"""

import argparse
import json
import os
import sys


def extract_pypdf(path):
    import pypdf
    reader = pypdf.PdfReader(path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text), len(reader.pages)


# Map 2-letter ISO codes → Tesseract 3-letter codes
_LANG_MAP = {"en": "eng", "id": "ind", "zh": "chi_sim", "ja": "jpn", "ko": "kor",
             "fr": "fra", "de": "deu", "es": "spa", "ar": "ara"}


def _normalize_lang(lang):
    return "+".join(_LANG_MAP.get(p.strip(), p.strip()) for p in lang.split("+"))


def extract_tesseract(path, lang="eng+ind"):
    import pytesseract
    from PIL import Image

    is_image = path.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"))
    lang = _normalize_lang(lang)

    if is_image:
        text = pytesseract.image_to_string(Image.open(path), lang=lang)
        return text, 1
    else:
        from pdf2image import convert_from_path
        images = convert_from_path(path)
        texts = [pytesseract.image_to_string(img, lang=lang) for img in images]
        return "\n".join(texts), len(images)


def main():
    parser = argparse.ArgumentParser(description="OCR PDF or image via Tesseract")
    parser.add_argument("--file", required=True, help="Path to PDF or image file")
    parser.add_argument("--lang", default="eng+ind",
                        help="Tesseract language(s), e.g. eng+ind (default: eng+ind)")
    parser.add_argument("--min-text", type=int, default=100,
                        help="Min pypdf chars before triggering OCR (default: 100)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(json.dumps({"error": f"File not found: {args.file}"}), file=sys.stderr)
        sys.exit(1)

    is_image = args.file.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"))

    # Fast path: pypdf for text-based PDFs
    if not is_image:
        try:
            text, pages = extract_pypdf(args.file)
            if len(text.strip()) >= args.min_text:
                print(json.dumps({"text": text, "pages": pages, "method": "pypdf", "chars": len(text)}))
                return
        except Exception as e:
            print(f"pypdf failed: {e}", file=sys.stderr)

    # Tesseract OCR fallback
    try:
        text, pages = extract_tesseract(args.file, args.lang)
        print(json.dumps({"text": text, "pages": pages, "method": "tesseract", "chars": len(text)}))
    except Exception as e:
        print(f"Tesseract failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
