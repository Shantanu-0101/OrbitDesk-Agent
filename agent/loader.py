"""
agent/loader.py
---------------
Loads and prepares all knowledge base documents and resolved cases
into a unified list of Document chunks ready for embedding and retrieval.

WHY THIS FILE EXISTS:
  Before the AI can search anything, it needs to READ and ORGANIZE
  all the source material. This loader acts like a librarian who reads
  every document and creates index cards before anyone searches the library.
"""

import os
import json
import re
from typing import List
from dataclasses import dataclass, field


# ── Data Structure 

@dataclass
class Document:
    """
    Represents a single searchable chunk of text.

    Each Document has:
      - source_id  : e.g. "KB-003" or "CASE-1041" (used in citations)
      - text       : the actual content to embed and search
      - metadata   : extra info like title, status, is_superseded
    """
    source_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ── Path Configuration 

# Walk up from this file to find the knowledge_base folder
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# These paths point to the assignment material (one level up from project)
_ASSIGNMENT_ROOT = os.path.dirname(_PROJECT_ROOT)
KB_DIR = os.path.join(_ASSIGNMENT_ROOT, "knowledge_base")
CASES_FILE = os.path.join(_ASSIGNMENT_ROOT, "resolved_cases.json")


# ── Helpers 

def _extract_document_id(content: str, filename: str) -> str:
    """
    Pull the document_id from the YAML front-matter of a markdown file.
    Example front-matter:
        ---
        document_id: KB-003
        ---
    If not found, fall back to the filename.
    """
    match = re.search(r"document_id:\s*(\S+)", content)
    if match:
        return match.group(1)
    # Fallback: use filename without extension
    return os.path.splitext(filename)[0].upper()


def _chunk_text(text: str, source_id: str, metadata: dict,
                chunk_size: int = 500, overlap: int = 50) -> List[Document]:
    """
    Split a long text into overlapping chunks.

    WHY CHUNKING?
      Large documents have many topics. If we embed the whole doc as one
      vector, the search loses precision. By splitting into ~500-character
      chunks with a 50-char overlap, we find the *specific* passage that
      answers the question — not just "this document is kinda relevant".

    The overlap prevents losing context at chunk boundaries.
    """
    # Clean up excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # If short enough, return as single chunk
    if len(text) <= chunk_size:
        return [Document(source_id=source_id, text=text, metadata=metadata)]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence or paragraph boundary
        if end < len(text):
            # Look backwards for a good break point (newline or period)
            break_point = text.rfind("\n", start, end)
            if break_point == -1 or break_point <= start:
                break_point = text.rfind(". ", start, end)
            if break_point > start:
                end = break_point + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_meta = {**metadata, "chunk_index": chunk_index}
            chunks.append(Document(
                source_id=source_id,
                text=chunk_text,
                metadata=chunk_meta
            ))
            chunk_index += 1

        start = end - overlap  # overlap keeps context between chunks

    return chunks


# ── Main Loaders 

def load_knowledge_base() -> List[Document]:
    """
    Load all markdown files from the knowledge_base/ directory.

    Each .md file gets:
      1. Its document_id extracted from YAML front-matter (e.g. KB-003)
      2. YAML front-matter stripped (we only want the actual content)
      3. Split into overlapping chunks for precise retrieval

    Returns a list of Document objects.
    """
    documents = []

    if not os.path.isdir(KB_DIR):
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")

    md_files = sorted([f for f in os.listdir(KB_DIR) if f.endswith(".md")])

    for filename in md_files:
        filepath = os.path.join(KB_DIR, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Extract document ID from front-matter
        doc_id = _extract_document_id(raw_content, filename)

        # Strip YAML front-matter (everything between the first two ---)
        content = re.sub(r"^---.*?---\s*", "", raw_content, flags=re.DOTALL).strip()

        # Extract title from front-matter for metadata
        title_match = re.search(r"title:\s*(.+)", raw_content)
        title = title_match.group(1).strip() if title_match else filename

        metadata = {
            "source_type": "knowledge_base",
            "filename": filename,
            "title": title,
            "doc_id": doc_id,
            "is_superseded": False,
        }

        chunks = _chunk_text(content, source_id=doc_id, metadata=metadata)
        documents.extend(chunks)

        print(f"  ✅ Loaded KB doc: {doc_id} ({filename}) → {len(chunks)} chunk(s)")

    return documents


def load_resolved_cases() -> List[Document]:
    """
    Load all resolved support cases from resolved_cases.json.

    IMPORTANT RULES (from the assignment):
      - Cases marked 'superseded' are HISTORICAL and must NOT be presented
        as current guidance. We still load them (for testing retrieval) but
        flag them clearly so the verification node can catch misuse.
      - Current KB docs take precedence over resolved cases if they conflict.

    Each case is turned into a Document with a descriptive text block.
    """
    documents = []

    if not os.path.isfile(CASES_FILE):
        raise FileNotFoundError(f"Resolved cases file not found: {CASES_FILE}")

    with open(CASES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for case in data.get("cases", []):
        case_id = case.get("case_id", "UNKNOWN")
        status = case.get("status", "unknown")
        title = case.get("title", "")
        is_superseded = (status == "superseded")

        # Build a descriptive text block from the case fields
        symptoms_text = "\n".join(f"  - {s}" for s in case.get("symptoms", []))
        resolution_text = "\n".join(f"  - {r}" for r in case.get("resolution", []))
        superseded_note = ""
        if is_superseded:
            superseded_note = (
                f"\n⚠️ SUPERSEDED: {case.get('superseded_reason', '')} "
                f"Do NOT present this as current guidance."
            )

        text = (
            f"Case ID: {case_id} | Status: {status.upper()} | "
            f"Version: {case.get('product_version', 'N/A')}\n"
            f"Title: {title}\n"
            f"Symptoms:\n{symptoms_text}\n"
            f"Resolution:\n{resolution_text}"
            f"{superseded_note}"
        )

        # Add important_limit if present
        if "important_limit" in case:
            text += f"\nImportant Limit: {case['important_limit']}"

        metadata = {
            "source_type": "resolved_case",
            "case_id": case_id,
            "status": status,
            "is_superseded": is_superseded,
            "title": title,
        }

        documents.append(Document(
            source_id=case_id,
            text=text,
            metadata=metadata
        ))

        flag = "⚠️ SUPERSEDED" if is_superseded else "✅"
        print(f"  {flag} Loaded case: {case_id} — {title}")

    return documents


def load_all_documents() -> List[Document]:
    """
    Master loader: loads KB documents + resolved cases.
    Returns a single combined list used by the retriever.
    """
    print("\n📚 Loading knowledge base...")
    kb_docs = load_knowledge_base()

    print("\n📋 Loading resolved cases...")
    case_docs = load_resolved_cases()

    all_docs = kb_docs + case_docs
    print(f"\n✅ Total documents loaded: {len(all_docs)} chunks\n")
    return all_docs


# ── Quick test (run this file directly to verify) 
if __name__ == "__main__":
    docs = load_all_documents()
    print("\n── Sample document ──")
    print(f"Source ID : {docs[0].source_id}")
    print(f"Metadata  : {docs[0].metadata}")
    print(f"Text (first 200 chars):\n{docs[0].text[:200]}")
