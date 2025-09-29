# Regulatory QnA Generator (PoC)

Convert regulatory documents (PDF/HTML/TXT or URLs) into strict JSON QnA using LLMs. Includes dynamic chunking for full coverage, URL/PDF ingestion, and multiple providers (Groq, Ollama, Gemini, OpenAI). Appends results to an existing JSON file.

## Quick Start

### 1) Python env and dependencies (MAC)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Providers (choose one or more)

- Groq (default)
```bash
pip install groq
export GROQ_API_KEY=your_key
```

- Ollama (local; no rate limits)
```bash
brew install ollama
ollama serve
ollama pull mistral
```

- Gemini
```bash
pip install google-generativeai
export GOOGLE_API_KEY=your_key
```

- OpenAI
```bash
pip install openai
export OPENAI_API_KEY=your_key
```

## Usage

Basic (Groq default, llama-3.1-8b-instant):
```bash
python main.py \
  --input /absolute/path/to/document.pdf \
  --output /absolute/path/to/out.json
```

Ollama (Mistral):
```bash
python main.py \
  --input /absolute/path/to/document.pdf \
  --provider ollama \
  --model mistral \
  --output /absolute/path/to/out.json
```

Gemini:
```bash
python main.py \
  --input /absolute/path/to/document.pdf \
  --provider gemini \
  --model gemini-1.5-flash \
  --output /absolute/path/to/out.json
```

OpenAI:
```bash
python main.py \
  --input /absolute/path/to/document.pdf \
  --provider openai \
  --model gpt-4o-mini \
  --output /absolute/path/to/out.json
```

URL ingestion (HTML/PDF):
```bash
python main.py \
  --input https://website.rbi.org.in/en/web/rbi/-/notifications/master-circular-housing-finance-for-ucbs#_General \
  --provider ollama --model mistral \
  --output /absolute/path/to/out.json
```

## Features
- PDF/HTML/TXT and http/https URL ingestion
- Dynamic chunking (default): page-grouping or character chunks with overlap
- Strict JSON output with validation and de-duplication
- Appends to existing JSON file (list or single-object compatible)
- Citations include `chunk:X/Y` for traceability

## CLI Flags
```text
--input            File path or URL (required)
--output           Output JSON path (default: <input>_qna.json)
--provider         groq (default) | ollama | gemini | openai
--model            Provider model (default: llama-3.1-8b-instant)
--ollama-url       Ollama base URL (default: http://localhost:11434)
--chunk-mode       dynamic (default) | auto | page | char | off
--chunk-size       Character chunk size for char/page merging (default: 18000)
--chunk-overlap    Character overlap for char chunks (default: 1000)
```

## Tips
- For large docs, prefer `--chunk-mode dynamic` (default) or `--chunk-mode page` for PDFs.
- To avoid API limits, use Ollama (`--provider ollama --model mistral`).
- The script appends results to an existing output file; keep a single `--output` path to accumulate.

## Development
- Python: 3.10+
- Key libs: pypdf, beautifulsoup4, requests, google-generativeai, openai, groq

## Security
- Add secrets to `.env` and ensure `.env` is in `.gitignore` (already configured).
- If a secret leaked in Git, rotate it and remove the file from history.

## License
PoC / internal use.
