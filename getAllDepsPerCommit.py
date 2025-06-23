import json
import pandas as pd
import sys

prefix = sys.argv[1]

# Load merged dependency snapshot
with open(f"{prefix}_merged_dependencies.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Sort commits chronologically (optional but preferred)
# If you have commit dates: load them and sort accordingly
commit_order = list(data.keys())

# Holds last known dependency count per file
last_known_counts = {}

# Final output rows
rows = []

metrics_df = pd.read_csv(f"{prefix}_commits_dependencies.csv", usecols=["commit_hash", "commit_date"])
commit_to_date = dict(zip(metrics_df["commit_hash"], metrics_df["commit_date"]))

'''for commit in commit_order:
    if commit not in data:
        continue
    systems = data[commit]
    current_counts = {}

    # Update current file-level counts
    for ecosystem, files in systems.items():
        for file, deps in files.items():
            key = (ecosystem, file)
            current_counts[key] = len(set(deps.keys()))
            last_known_counts[key] = current_counts[key]  # update history

    # Fill in unchanged files from history
    full_file_counts = {
        key: last_known_counts[key] for key in last_known_counts
    }

    total_count = sum(full_file_counts.values())

    rows.append({
        "date": commit_to_date.get(commit, ""),
        "commit": commit,
        "total_unique_dependencies": total_count
    })
'''

for commit in commit_order:
    if commit not in commit_to_date:
        continue  # skip commits without a date

    systems = data[commit]
    all_deps = set()

    for ecosystem, files in systems.items():
        for file, deps in files.items():
            all_deps.update(deps.keys())  # add all dependency names

    rows.append({
        "date": commit_to_date[commit],
        "commit": commit,
        "dependencies": sorted(all_deps)  # optional: sort for consistency
    })

rows.sort(key=lambda x: x['date'])
# Save to CSV
df = pd.DataFrame(rows)
df.to_csv(f"{prefix}_dependency_count_per_commit_filled.csv", index=False)
