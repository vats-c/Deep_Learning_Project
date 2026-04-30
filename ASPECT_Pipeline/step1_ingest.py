import re
import json
import os
import logging
from pathlib import Path

from config import (
    MD_ROOT, CHUNKS_FILE, OUTPUTS_DIR,
    MAX_CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)
log = logging.getLogger(__name__)


def split_large_chunk(text: str, max_size: int = MAX_CHUNK_SIZE,
                       overlap: int = CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_size, len(text))
        chunks.append(text[start:end])
        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def chunk_markdown(filepath: str):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    heading_re = re.compile(r"(?=^\s*#{1,3}\s)", re.MULTILINE)
    sections = [s.strip() for s in heading_re.split(text) if s.strip()]

    merged, buffer = [], ""
    for section in sections:
        if len(buffer) + len(section) < MAX_CHUNK_SIZE:
            buffer += "\n\n" + section
        else:
            if buffer:
                merged.append(buffer.strip())
            buffer = section
    if buffer:
        merged.append(buffer.strip())

    final = []
    for chunk in merged:
        if len(chunk) > MAX_CHUNK_SIZE * 2:
            final.extend(split_large_chunk(chunk))
        else:
            final.append(chunk)

    return final


def format_chunks(raw_chunks: list, filepath: str):
    doc_name = os.path.basename(filepath)
    doc_id   = os.path.splitext(doc_name)[0].replace(" ", "_")
    structured = []

    for i, text in enumerate(raw_chunks):
        header_match  = re.search(r"^(#{1,3})\s+(.+)", text, re.MULTILINE)
        section_header = header_match.group(2).strip() if header_match else "General"

        structured.append({
            "document_name":    doc_name,
            "doc_id":           doc_id,
            "chunk_id":         f"{doc_id}_chunk_{i}",
            "chunk_index":      i,
            "section_header":   section_header,
            "section_breadcrumb": section_header,
            "char_length":      len(text),
            "text":             text.strip(),
        })

    return structured


def run():
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    md_files = list(Path(MD_ROOT).glob("*.md"))
    if not md_files:
        log.error(f"No .md files found in '{MD_ROOT}'.")
        return

    log.info(f"Found {len(md_files)} markdown file(s).")
    all_chunks = []

    for md_path in md_files:
        log.info(f"  Chunking: {md_path.name}")
        raw    = chunk_markdown(str(md_path))
        chunks = format_chunks(raw, str(md_path))
        all_chunks.extend(chunks)
        log.info(f"    → {len(chunks)} chunks")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(all_chunks)} total chunks → {CHUNKS_FILE}")


if __name__ == "__main__":
    run()