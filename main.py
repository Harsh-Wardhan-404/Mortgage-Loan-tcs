import argparse
import json
import mimetypes
import os
import re
import sys
from io import BytesIO
from urllib.parse import urlparse
from typing import Any, Dict, List

# Optional imports are guarded so the script can run with partial features
try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

import requests


def read_file_text(file_path: str) -> str:
    """Extract text from PDF, HTML, or plain text files.

    Falls back to reading as UTF-8 text if type is unknown.
    """
    # URL short-circuit
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return read_url_text(file_path)

    content_type, _ = mimetypes.guess_type(file_path)
    file_extension = (os.path.splitext(file_path)[1] or "").lower()

    if file_extension == ".pdf" or (content_type and "pdf" in content_type):
        if PdfReader is None:
            raise RuntimeError("pypdf is required to read PDFs. Install with: pip install pypdf")
        text_segments: List[str] = []
        reader = PdfReader(file_path)
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text.strip():
                text_segments.append(f"[Page {page_index + 1}]\n{page_text}")
        return "\n\n".join(text_segments)

    if file_extension in [".html", ".htm"] or (content_type and "html" in content_type):
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 is required to read HTML. Install with: pip install beautifulsoup4")
        with open(file_path, "rb") as file_handle:
            raw_bytes = file_handle.read()
        soup = BeautifulSoup(raw_bytes, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return normalize_text(text)

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file_handle:
        return normalize_text(file_handle.read())


def read_url_text(url: str) -> str:
    """Fetch a URL and extract text from HTML or PDF responses."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    if "application/pdf" in content_type or url.lower().endswith(".pdf"):
        if PdfReader is None:
            raise RuntimeError("pypdf is required to read PDFs. Install with: pip install pypdf")
        reader = PdfReader(BytesIO(response.content))
        text_segments: List[str] = []
        for page_index, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if page_text.strip():
                text_segments.append(f"[Page {page_index + 1}]\n{page_text}")
        return "\n\n".join(text_segments)

    if "text/html" in content_type or url.lower().endswith((".html", ".htm")):
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 is required to read HTML. Install with: pip install beautifulsoup4")
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return normalize_text(text)

    # Fallback: try to decode as text
    try:
        return normalize_text(response.text)
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def build_prompt(document_text: str, source_path: str, max_chars: int = 24000) -> str:
    """Construct the instruction prompt to obtain strict JSON QnA."""
    text = document_text
    if len(text) > max_chars:
        head = text[: max_chars // 2]
        tail = text[-max_chars // 2 :]
        text = head + "\n...\n" + tail

    prompt = f"""
You are a compliance analyst. Convert the following regulatory document into a structured QnA JSON.
- Focus on enforceable obligations, eligibility, disclosures, KYC/AML, risk, timelines, penalties, exceptions, and any mortgage-related clauses.
- Use clear, concise, domain-accurate language suitable for downstream automation.
- Cite sections/pages verbatim where possible for each QnA item.

Output MUST be a single JSON object with this schema ONLY:
{{
  "metadata": {{
    "source_path": "string",
    "doc_type": "regulatory",
    "language": "string|detected",
    "notes": "string"
  }},
  "qna": [
    {{
      "question": "string",
      "answer": "string",
      "citations": ["string"]
    }}
  ]
}}

Rules:
- Return STRICT, VALID JSON. No markdown, no comments, no trailing commas.
- Minimum 12 high-signal QnA pairs; more if material warrants (cap at 60).
- Consolidate duplicates; split multi-part obligations into separate QnAs.
- If parts of the text are truncated/unclear, include a caveat in "metadata.notes".

Document (source: {source_path}):
<document>
{text}
</document>
"""
    return prompt.strip()


def build_chunk_prompt(chunk_text: str, source_path: str, chunk_index: int, total_chunks: int) -> str:
    header = f"Chunk {chunk_index + 1} of {total_chunks}. Focus ONLY on this chunk; do not invent content beyond what is present here."
    base = build_prompt(chunk_text, source_path)
    return f"{header}\n\n{base}"


def has_page_markers(text: str) -> bool:
    return bool(re.search(r"^\[Page\s+\d+\]$", text.split("\n", 1)[0])) or "[Page " in text


def split_text_into_chunks(text: str, mode: str, chunk_size: int, overlap: int) -> List[str]:
    if mode == "dynamic":
        # dynamic acts like auto at text level; the detailed dynamic logic is applied at PDF source level below
        mode = "auto"
    if mode == "page" or (mode == "auto" and has_page_markers(text)):
        # Split by page marker lines: [Page N]
        pages: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if line.startswith("[Page ") and line.endswith("]"):
                if current:
                    pages.append("\n".join(current).strip())
                    current = []
            current.append(line)
        if current:
            pages.append("\n".join(current).strip())
        # Merge small pages into chunks near chunk_size
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for p in pages:
            if buf_len + len(p) + 1 > chunk_size and buf:
                chunks.append("\n\n".join(buf))
                buf = []
                buf_len = 0
            buf.append(p)
            buf_len += len(p) + 2
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

    # Character-based splitting with overlap
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        # try to end on a paragraph boundary if possible
        if end < n:
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.6:
                end = start + last_break
                chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= n:
            break
        start = max(end - overlap, 0)
    return [c for c in chunks if c]


def get_dynamic_chunking_strategy_for_path(source: str) -> Dict[str, Any]:
    """Decide chunking based on PDF page count or file size for local files.

    For URLs or non-files, fallback to character-based strategy.
    """
    try:
        file_size_mb = os.path.getsize(source) / (1024 * 1024) if os.path.exists(source) else None
    except Exception:
        file_size_mb = None

    page_count = None
    if source.lower().endswith('.pdf') and os.path.exists(source):
        try:
            if PdfReader is not None:
                reader = PdfReader(source)
                page_count = len(reader.pages)
        except Exception:
            page_count = None

    if page_count is not None:
        if page_count < 20:
            return {"mode": "page", "page_group": 1, "overlap_pages": 0, "char_size": 14000, "char_overlap": 800}
        if page_count < 100:
            return {"mode": "page", "page_group": 3, "overlap_pages": 1, "char_size": 16000, "char_overlap": 1000}
        return {"mode": "char", "page_group": 0, "overlap_pages": 0, "char_size": 18000, "char_overlap": 1200}

    # Fallback by file size (or URL/no size)
    if file_size_mb is None or file_size_mb < 1:
        return {"mode": "char", "page_group": 0, "overlap_pages": 0, "char_size": 12000, "char_overlap": 800}
    if file_size_mb < 10:
        return {"mode": "char", "page_group": 0, "overlap_pages": 0, "char_size": 16000, "char_overlap": 1000}
    return {"mode": "char", "page_group": 0, "overlap_pages": 0, "char_size": 20000, "char_overlap": 1200}


def group_pages_by_count(text: str, group_size: int, overlap_pages: int) -> List[str]:
    pages: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.startswith("[Page ") and line.endswith("]"):
            if current:
                pages.append("\n".join(current).strip())
                current = []
        current.append(line)
    if current:
        pages.append("\n".join(current).strip())

    if not pages:
        return [text]

    chunks: List[str] = []
    i = 0
    n = len(pages)
    while i < n:
        j = min(i + group_size, n)
        group = pages[i:j]
        chunks.append("\n\n".join(group))
        if j >= n:
            break
        i = max(j - overlap_pages, 0)
    return chunks


def call_ollama(model_name: str, prompt: str, base_url: str = "http://localhost:11434") -> str:
	url = f"{base_url}/api/chat"
	payload = {
		"model": model_name,
		"messages": [{"role": "user", "content": prompt}],
		"options": {"temperature": 0.2},
		"format": "json",
		"stream": False  # ensure a single JSON response
	}
	resp = requests.post(url, json=payload, timeout=600)
	resp.raise_for_status()
	try:
		data = resp.json()
	except requests.exceptions.JSONDecodeError:
		# Fallback: handle NDJSON (streamed) response
		content = ""
		for line in resp.text.splitlines():
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
				# Ollama stream lines often contain 'message': {'content': '...'} fragments
				if "message" in obj and obj["message"].get("content"):
					content += obj["message"]["content"]
				elif "response" in obj:
					content += obj["response"]
			except Exception:
				continue
		if not content.strip():
			raise
		return content

	# Non-streaming path
	if "message" in data and "content" in data["message"]:
		return data["message"]["content"]
	if "response" in data:
		return data["response"]
	raise RuntimeError("Unexpected Ollama response format")


def call_openai(model_name: str, prompt: str) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        raise RuntimeError("openai SDK not installed. Install with: pip install openai")
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return completion.choices[0].message.content  # type: ignore


def call_gemini(model_name: str, prompt: str) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        raise RuntimeError("google-generativeai not installed. Install with: pip install google-generativeai")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }
    model = genai.GenerativeModel(model_name=model_name, generation_config=generation_config)
    result = model.generate_content(prompt)
    if not result or not getattr(result, "text", ""):
        raise RuntimeError("Empty response from Gemini.")
    return result.text  # already JSON per response_mime_type


def call_groq(model_name: str, prompt: str) -> str:
    try:
        from groq import Groq  # type: ignore
    except Exception:
        raise RuntimeError("groq not installed. Install with: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def ensure_json(json_like_text: str) -> Dict[str, Any]:
    clean_text = json_like_text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(json)?\s*", "", clean_text)
        clean_text = re.sub(r"\s*```$", "", clean_text)
    parsed: Dict[str, Any] = json.loads(clean_text)

    if "metadata" not in parsed or "qna" not in parsed:
        raise ValueError("JSON missing required keys: 'metadata' and/or 'qna'")
    if not isinstance(parsed["qna"], list):
        raise ValueError("'qna' must be a list")

    seen_questions = set()
    unique_qna: List[Dict[str, Any]] = []
    for item in parsed["qna"]:
        if not isinstance(item, dict):
            continue
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        citations = item.get("citations") or []
        if not question or not answer:
            continue
        key = question.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        if not isinstance(citations, list):
            citations = [str(citations)]
        unique_qna.append({
            "question": question,
            "answer": answer,
            "citations": [str(c) for c in citations],
        })
    parsed["qna"] = unique_qna
    return parsed


def detect_language(sample_text: str) -> str:
    snippet = sample_text[:1000]
    if re.search(r"[ऀ-ॿ]", snippet):
        return "hi"
    if re.search(r"[A-Za-z]", snippet):
        return "en"
    return "unknown"


def _default_output_path_for_source(source: str) -> str:
    if source.startswith("http://") or source.startswith("https://"):
        parsed = urlparse(source)
        safe_netloc = re.sub(r"[^A-Za-z0-9_.-]", "_", parsed.netloc)
        safe_path = re.sub(r"[^A-Za-z0-9_.-]", "_", parsed.path.strip("/")) or "root"
        base = f"{safe_netloc}_{safe_path}_qna.json"
        return os.path.abspath(base)
    return os.path.abspath(os.path.splitext(source)[0] + "_qna.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate regulatory QnA JSON from a document.")
    parser.add_argument("--input", required=True, help="Path to input file (PDF/HTML/TXT).")
    parser.add_argument("--output", required=False, help="Path to output JSON file.")
    parser.add_argument(
        "--provider",
        choices=["gemini", "ollama", "openai", "groq"],
        default="groq",
        help="LLM provider.",
    )
    parser.add_argument(
        "--model",
        default="llama-3.1-8b-instant",
        help="Model name. e.g., 'llama-3.1-8b-instant' (Groq), 'gemini-1.5-flash' (Gemini), 'mistral' (Ollama), or 'gpt-4o-mini' (OpenAI).",
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Base URL for Ollama.")
    parser.add_argument("--chunk-mode", choices=["dynamic", "auto", "page", "char", "off"], default="dynamic", help="Chunking strategy.")
    parser.add_argument("--chunk-size", type=int, default=18000, help="Chunk size in characters for char/page merging.")
    parser.add_argument("--chunk-overlap", type=int, default=1000, help="Character overlap between chunks (char mode).")
    args = parser.parse_args()

    input_arg = args.input
    is_url = input_arg.startswith("http://") or input_arg.startswith("https://")
    input_source = input_arg if is_url else os.path.abspath(input_arg)
    if not is_url and not os.path.exists(input_source):
        print(f"Input file not found: {input_source}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or _default_output_path_for_source(input_source)
    output_path = os.path.abspath(output_path)

    print(f"Reading: {input_source}")
    document_text = read_file_text(input_source)
    if not document_text.strip():
        print("No extractable text found.", file=sys.stderr)
        sys.exit(2)

    language_code = detect_language(document_text)
    # Chunking
    if args.chunk_mode == "off":
        chunks = [document_text]
    elif args.chunk_mode == "dynamic":
        strategy = get_dynamic_chunking_strategy_for_path(input_source)
        if strategy["mode"] == "page" and has_page_markers(document_text):
            chunks = group_pages_by_count(document_text, strategy["page_group"], strategy["overlap_pages"])
        else:
            chunks = split_text_into_chunks(document_text, "char", strategy["char_size"], strategy["char_overlap"])
        if not chunks:
            chunks = [document_text]
        print(f"Dynamic chunking -> mode={strategy['mode']}, chunks={len(chunks)}")
    else:
        chunks = split_text_into_chunks(document_text, args.chunk_mode, args.chunk_size, args.chunk_overlap)
        if not chunks:
            chunks = [document_text]

    aggregated_qna: List[Dict[str, Any]] = []
    notes: List[str] = []
    total_chunks = len(chunks)
    for idx, chunk in enumerate(chunks):
        chunk_prompt = build_chunk_prompt(chunk, input_source, idx, total_chunks)
        print(f"Generating QnA for chunk {idx + 1}/{total_chunks} ...")
        if args.provider == "ollama":
            raw_response = call_ollama(args.model, chunk_prompt, base_url=args.ollama_url)
        elif args.provider == "openai":
            raw_response = call_openai(args.model, chunk_prompt)
        elif args.provider == "groq":
            raw_response = call_groq(args.model, chunk_prompt)
        else:
            raw_response = call_gemini(args.model, chunk_prompt)

        try:
            chunk_obj = ensure_json(raw_response)
        except Exception:
            match = re.search(r"\{[\s\S]*\}\s*$", raw_response)
            if match:
                try:
                    chunk_obj = ensure_json(match.group(0))
                except Exception:
                    notes.append(f"Chunk {idx + 1}: failed to parse JSON; content skipped")
                    continue
            else:
                notes.append(f"Chunk {idx + 1}: failed to parse JSON; content skipped")
                continue

        for item in chunk_obj.get("qna", []):
            # Tag citations with chunk info to aid traceability
            citations = item.get("citations") or []
            if not isinstance(citations, list):
                citations = [str(citations)]
            citations = [str(c) for c in citations]
            citations.append(f"chunk:{idx + 1}/{total_chunks}")
            aggregated_qna.append({
                "question": (item.get("question") or "").strip(),
                "answer": (item.get("answer") or "").strip(),
                "citations": citations,
            })

    # Deduplicate across chunks using the same logic as ensure_json
    seen_questions = set()
    merged_qna: List[Dict[str, Any]] = []
    for it in aggregated_qna:
        q = it.get("question", "").strip()
        a = it.get("answer", "").strip()
        if not q or not a:
            continue
        key = q.lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        merged_qna.append(it)

    json_obj = {
        "metadata": {
            "source_path": input_source,
            "doc_type": "regulatory",
            "language": language_code,
            "notes": "; ".join(notes),
        },
        "qna": merged_qna,
    }

    # Append behavior: if file exists, append; if single object, convert to list then append
    existing: Any = None
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = None

    if existing is None:
        # Create a new file. Use a list to support future appends seamlessly.
        payload: Any = json_obj
    else:
        if isinstance(existing, list):
            existing.append(json_obj)
            payload = existing
        elif isinstance(existing, dict):
            payload = [existing, json_obj]
        else:
            payload = [json_obj]

    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    print(f"Wrote QnA JSON: {output_path}")


if __name__ == "__main__":
    main()