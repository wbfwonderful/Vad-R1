import os
import json

output_lines = []

for filename in os.listdir("./"):
    if filename.endswith(".jsonl"):
        total_reasonability = 0.0
        total_detail = 0.0
        total_consistency = 0.0
        count = 0

        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    output = json.loads(data["output"])

                    total_reasonability += output.get("reasonability", 0.0)
                    total_detail += output.get("detail", 0.0)
                    total_consistency += output.get("consistency", 0.0)
                    count += 1
                except Exception as e:
                    print(f"Error in file {filename}: {e}")

        if count > 0:
            avg_r = total_reasonability / count
            avg_d = total_detail / count
            avg_c = total_consistency / count
            line = f"{filename}\tReasonability: {avg_r:.3f}, Detail: {avg_d:.3f}, Consistency: {avg_c:.3f}"
        else:
            line = f"{filename}\tNo valid data."

        output_lines.append(line)

with open("results.txt", "w", encoding="utf-8") as out_file:
    for line in output_lines:
        out_file.write(line + "\n")

