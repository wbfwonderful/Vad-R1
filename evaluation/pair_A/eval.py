import json
import re
import os
import csv

pattern = re.compile(r"\[\[([ABC])\]\]")

jsonl_files = [f for f in os.listdir(".") if f.endswith(".jsonl")]

results = []

for jsonl_path in jsonl_files:
    a_win, b_win, tie = 0, 0, 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            output = json.loads(line).get("output", "")
            match = pattern.search(output)
            if match:
                verdict = match.group(1)
                if verdict == "A":
                    a_win += 1
                elif verdict == "B":
                    b_win += 1
                elif verdict == "C":
                    tie += 1

    total = a_win + b_win + tie
    a_win_rate = a_win / total * 100 if total else 0
    b_win_rate = b_win / total * 100 if total else 0
    tie_rate = tie / total * 100 if total else 0

    result = {
        "file": jsonl_path.split("/")[-1].replace(".jsonl", ""),
        "A_win_rate(%)": round(a_win_rate, 2),
        "B_win_rate(%)": round(b_win_rate, 2),
        "Tie_rate(%)": round(tie_rate, 2)
    }
    results.append(result)

results.sort(key=lambda x: x["B_win_rate(%)"], reverse=True)

with open("results.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

