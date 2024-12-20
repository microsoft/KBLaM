import pandas as pd
import json


def flatten_results(data):
    rows = []
    for exp in data:
        kb_size = exp["kb_size"]
        for result in exp["accuracy_results"]:
            rows.append(
                {
                    "kb_size": kb_size,
                    "idx": result["idx"],
                    "acc": result["acc"],
                    "top5acc": result["top5acc"],
                }
            )
    return rows


# Load JSON from file
with open("/datadisk/kblam_attention_acc_results/accuracy_results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame and save as CSV
df = pd.DataFrame(flatten_results(results))
df = df.sort_values(["kb_size", "idx"]).reset_index(drop=True)

# Save to CSV
df.to_csv("accuracy_results.csv", index=False)
