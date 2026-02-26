from pathlib import Path
import json

from llama_cpp import Llama

MODEL_PATH = "models/gemma-3-270m-q8_0.gguf"
print("CWD:", Path.cwd())
print("MODEL_PATH:", MODEL_PATH)
print("MODEL_EXISTS:", Path(MODEL_PATH).is_file())
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

def main():
    print("Loading Gemma model...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=2,   # GitHub runners have ~2 cores
        n_batch=128,
        verbose=False,
    )

    prompt = build_prompt()

    print("Running inference...")
    output = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["\n\n\n", "</s>"],
    )

    text = output["choices"][0]["text"].strip()
    print("=== RAW MODEL OUTPUT ===")
    print(text)

    # Try to parse JSON
    try:
        data = json.loads(text)
        print("\n=== Parsed JSON ===")
        print(json.dumps(data, indent=2))
        # Optional: enforce minimum correctness score
        if data.get("correctness_score", 0) < 6:
            print("\nAssignment did not meet minimum correctness threshold.")
            exit(1)
    except json.JSONDecodeError:
        print("\nCould not parse model output as JSON; leaving job as success.")
        # You may choose to `exit(1)` here if you want to fail on malformed output.

if __name__ == "__main__":
    main()
