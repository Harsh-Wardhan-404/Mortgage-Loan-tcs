import json
import argparse
from typing import Any, List, Dict


def is_error_only(entry: Dict[str, Any]) -> bool:
	# Remove entries with empty qna
	qna = entry.get("qna", [])
	if not qna:
		return True
	# Remove entries whose notes indicate model-not-found or all failed parses
	notes: str = entry.get("metadata", {}).get("notes", "") or ""
	bad_markers = [
		"model_not_found",
		"does not exist",
		"failed to parse JSON",
		"provider error",
	]
	return all(marker in notes for marker in bad_markers[:1]) or ("failed to parse JSON" in notes and "Chunk" in notes and len(qna) == 0)


def main() -> None:
	parser = argparse.ArgumentParser(description="Clean QnA JSON: remove empty or error-only entries")
	parser.add_argument("--input", required=True, help="Path to aggregated QnA JSON")
	parser.add_argument("--output", required=False, help="Path to write cleaned JSON (defaults to overwrite input)")
	args = parser.parse_args()

	with open(args.input, "r", encoding="utf-8") as fh:
		data = json.load(fh)

	# Normalize to list
	if isinstance(data, dict):
		entries: List[Dict[str, Any]] = [data]
	else:
		entries = list(data)

	cleaned = [e for e in entries if not is_error_only(e)]

	out_path = args.output or args.input
	with open(out_path, "w", encoding="utf-8") as fh:
		json.dump(cleaned, fh, ensure_ascii=False, indent=2)
	print(f"Cleaned entries: kept {len(cleaned)} of {len(entries)} → {out_path}")


if __name__ == "__main__":
	main()






