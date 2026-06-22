import json
from pathlib import Path

src = Path("export/eval/eval_questions.json")
out = Path("export/eval/eval_questions.jsonl")

data = json.loads(src.read_text())
questions = data["questions"]

with open(out, "w") as f:
    for q in questions:
        f.write(json.dumps(q) + "\n")

print(f"Wrote {len(questions)} questions to {out}")
