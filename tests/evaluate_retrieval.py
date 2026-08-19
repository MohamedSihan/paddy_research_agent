import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import EvidenceRetriever

bench = json.loads(Path(__file__).with_name("retrieval_eval.json").read_text())
r = EvidenceRetriever()
hits = 0
for case in bench:
    got = [x["id"] for x in r.search(case["query"], 5)]
    expected = set(case["relevant_ids"])
    found = len(expected.intersection(got))
    p5 = found / 5
    hits += p5
    print(f"{case['id']}: Precision@5={p5:.2f} | expected={list(expected)} | got={got}")
print(f"Mean Precision@5: {hits/len(bench):.2f}")
