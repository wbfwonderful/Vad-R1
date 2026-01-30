import os
import json
import pandas as pd

def compute_double_right_stats(jsonl_path, threshold=0.8):
    RR = RW = WR = WW = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            pred = item["pred_type"]
            gt = item["anomaly_type"]
            sim = item["similarity"]

            if pred == "Normal" and gt == "Normal":
                pred_right = True
            elif pred == "Abnormal" and gt != "Normal":
                pred_right = True
            else:
                pred_right = False

            rationale_right = (sim >= threshold)

            if pred_right and rationale_right:
                RR += 1
            elif pred_right and not rationale_right:
                RW += 1
            elif not pred_right and rationale_right:
                WR += 1
            else:
                WW += 1
    
    total = RR + RW + WR + WW
    if total == 0:
        return {"RR": 0.0, "RW": 0.0, "WR": 0.0, "WW": 0.0}

    p_rr = RR / total * 100
    p_rw = RW / total * 100
    p_wr = WR / total * 100
    rr = round(p_rr, 2)
    rw = round(p_rw, 2)
    wr = round(p_wr, 2)
    ww = round(100.00 - rr - rw - wr, 2)

    return {"RR": rr, "RW": rw, "WR": wr, "WW": ww}


def evaluate_all_jsonl_files(folder_path=".", threshold=0.8,
                             output_json="./qwen_summary.json",
                             output_excel="./qwen_summary_table.xlsx"):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jsonl") and filename.startswith("double_right_"):
            full_path = os.path.join(folder_path, filename)
            stats = compute_double_right_stats(full_path, threshold)
            model_name = filename.replace("double_right_", "").replace(".jsonl", "")
            results[model_name] = stats


    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.name = "Model"
    df = df[["RR", "RW", "WR", "WW"]]
    df = df.sort_values(by="RR", ascending=False)


    print(df.to_string(float_format="%.2f"))

    df.to_excel(output_excel)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved Excel table to {output_excel}")
    print(f"✅ Saved JSON results to {output_json}")


if __name__ == "__main__":
    evaluate_all_jsonl_files("./", threshold=0.85)
