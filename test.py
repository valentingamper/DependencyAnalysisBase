import subprocess
from typing import List, Tuple
import pygit2

patterns = [
    "requirements.txt"
]



def clone_repository(repo_name: str) -> str:
    repo_url = "https://github.com/" + repo_name
    target_dir = "temp/" + repo_name
    subprocess.run([
        "git", "clone",
        "--bare",
        "--single-branch",
        repo_url,
        target_dir
    ], check=True)

    return target_dir


def get_commits_where_patterns_got_touched(target_dir: str) -> List[str]:
    fmt = "%H%x00%ct%x00"

    result = subprocess.run([
                                "git",
                                "--git-dir",
                                str(target_dir),
                                "log",
                                f"--pretty=format:{fmt}",
                                "--"
                            ] + patterns, capture_output=True, text=True)

    if result.stdout is None:
        raise Exception("No commits found")

    parts = [p for p in result.stdout.replace("\n", "").split("\x00") if p]
    return [c for c, _ in sorted(list([(c, int(ts)) for c, ts in zip(parts[0::2], parts[1::2])]), key=lambda x: x[1])]


#s = "valentingamper/DependencyAnalysisDeletedTest"
s = "microsoft/BitNet"
target = clone_repository(s)
results = get_commits_where_patterns_got_touched(target)

repo = pygit2.Repository(target)

for commit_hash in results:
    commit = repo.revparse_single(results[0])
    diff = None

    if commit.parents:
        diff = repo.diff(commit.parents[0], commit)
    else:
        diff = commit.tree.diff_to_tree()

    for patch in diff:
        if any(patch.delta.old_file.path.endswith(e) for e in patterns) or any(patch.delta.new_file.path.endswith(e) for e in patterns):
            print(f"File: {patch.delta.old_file.path} -> {patch.delta.new_file.path}")
