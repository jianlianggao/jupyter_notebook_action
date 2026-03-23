from pathlib import Path
import json, re, csv

from llama_cpp import Llama

MODEL_PATH = "models/Qwen2.5-1.5B-Instruct-Q4_0.gguf"
print("CWD:", Path.cwd())
print("MODEL_PATH:", MODEL_PATH)
print("MODEL_EXISTS:", Path(MODEL_PATH).is_file())

SUMMARY_PATH = Path("results/llama_summary.json")
SUMMARY_CSV_PATH = Path("results/llama_summary.csv")

def collect_code(max_chars=6000):
    files = []
    for pattern in ["*.py", "*.ipynb"]:
        files.extend(Path(".").glob(pattern))
    # Ignore baseline/tests/results/.github
    files = [f for f in files if not str(f).startswith(("baseline", "tests", "results", ".github"))]

    chunks = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + "\n# [TRUNCATED]"
        chunks.append(f"--- FILE: {f} ---\n{text}")
    return "\n\n".join(chunks) if chunks else "No student code found."
def is_heading_only_markdown(text: str) -> bool:
    """
    Return True if the markdown text consists only of heading lines
    (lines starting with #) and/or blank lines.
    """
    lines = text.splitlines()
    has_nonempty_line = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            # blank line
            continue
        has_nonempty_line = True
        # line must start with one or more '#' to be considered heading
        if not stripped.startswith("#"):
            return False

    # If there are no non-empty lines, treat as not heading-only (you can decide)
    if not has_nonempty_line:
        return False

    return True

def strip_leading_headings(text: str) -> str:
    """
    Remove leading heading lines (#...) at the top of the cell,
    but keep the rest of the content.
    """
    lines = text.splitlines()
    new_lines = []
    non_heading_seen = False

    for line in lines:
        stripped = line.strip()
        if not non_heading_seen and stripped.startswith("#"):
            # skip leading heading lines
            continue
        else:
            non_heading_seen = True
            new_lines.append(line)

    return " ".join(new_lines)

def extract_markdown_from_notebook_clean(nb_path: str):
    nb_path = Path(nb_path)
    with nb_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    markdown_cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue

        raw_text = "".join(cell.get("source", []))
        if len(raw_text.strip()) == 0:
            continue

        # If the entire cell is only headings, skip it
        if is_heading_only_markdown(raw_text):
            continue

        # Otherwise, strip heading lines at the top and keep the prose
        cleaned = strip_leading_headings(raw_text).strip()
        if cleaned:
            markdown_cells.append(cleaned)

    return markdown_cells

def text_is_too_short(text: str, min_words: int = 50) -> bool:
    """
    Return True if the text is considered too short to meaningfully evaluate.
    """
    if isinstance(text, list):
        text = "".join(text)
    text = str(text).strip()
    if len(text) < min_words:
        return True
    return False

def evaluate_text_plain(text):
    # Make sure text is a string, not ['...']
    if isinstance(text, list):
        text = "".join(text)
    text = str(text)
    
    prompt = f"""
You are a teacher. Read the following student text and briefly evaluate it
in terms of content, organization, language use, and mechanics.

Write 4–6 sentences in total.

Text:
{text}
"""
    print("----- prompt -----")
    print(prompt)
    print("------------------")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=131072,
        n_threads=2,   # GitHub runners have ~2 cores
        n_batch=128,
        verbose=False,
    )
    resp = llm(prompt, 
               max_tokens=256, 
               temperature=0.7,
               top_k=20,
               top_p=0.9
               )
    raw = resp["choices"][0]["text"].strip()
    print("----- raw output -----")
    print(repr(raw))
    print("----------------------")
    return raw

def build_prompt():
    
    code = collect_code()
    instructions = """
You are grading a Python assignment.

You will see student code (truncated if needed).

Return JSON only with this exact schema:
{
  "correctness_score": 0-10,
  "style_score": 0-10,
  "key_findings": ["short bullet points"],
  "overall_feedback": "2-5 sentences of feedback"
}

Be strict but fair. Do not add any text outside the JSON.
"""
    return instructions + "\n\nSTUDENT CODE:\n\n" + code

def evaluate_single_notebook(nb_path: Path):
    """
    Evaluate one notebook and return a structured record.
    """
    print(f"\n=== Processing notebook: {nb_path} ===")
    markdown_cells = extract_markdown_from_notebook_clean(str(nb_path))

    if not markdown_cells:
        print("[INFO] No markdown content found.")
        return {
            "notebook": str(nb_path),
            "status": "no_markdown",
            "evaluation": "",
            "error": None,
        }
    if text_is_too_short(markdown_cells):
        print("[INFO] Text is too short, skipping LLM evaluation.")
        return {
            "notebook": str(nb_path),
            "status": "too_short",
            "evaluation": "[TOO_SHORT] The text is too short to evaluate meaningfully.",
            "error": None,
        }
    try:
        eval_text = evaluate_text_plain(markdown_cells)
        print("EVALUATION:", eval_text)
        return {
            "notebook": str(nb_path),
            "status": "ok",
            "evaluation": eval_text,
            "error": None,
        }
    except Exception as e:
        print(f"[ERROR] Evaluation failed for {nb_path}: {e}")
        return {
            "notebook": str(nb_path),
            "status": "error",
            "evaluation": "",
            "error": repr(e),
        }


def evaluate_multiple_notebooks(folder: str):
    folder = Path(folder)
    all_results = []

    for nb_path in sorted(folder.glob("test*.ipynb")):
        print(f"\n=== Processing notebook: {nb_path} ===")
        nb_results = evaluate_single_notebook(str(nb_path))
        all_results.append(nb_results)

    return all_results

def main():
    print("Loading Llama model...")
    
    results = evaluate_multiple_notebooks(Path.cwd())
    #SUMMARY_PATH.parent.mkdir(exist_ok=True)
    #with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    #    json.dump(results, f, indent=2, ensure_ascii=False)

    #print(f"[INFO] Wrote summary to {SUMMARY_PATH.resolve()}")
    SUMMARY_CSV_PATH.parent.mkdir(exist_ok=True)
    fieldnames = ["notebook", "status", "evaluation", "error"]
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Ensure only known keys are written
            writer.writerow({
                "notebook": row.get("notebook", ""),
                "status": row.get("status", ""),
                "evaluation": row.get("evaluation", ""),
                "error": row.get("error", ""),
            })

    print(f"[INFO] Wrote Llama evaluation CSV to {SUMMARY_CSV_PATH.resolve()}")

if __name__ == "__main__":
    main()
