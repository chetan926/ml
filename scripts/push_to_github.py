"""
GitHub Secure Publication Script.
Pushes branch 'main' to https://github.com/chetan926/ml.git using GitHub Personal Access Token.
"""

import os
import sys
import getpass
from pathlib import Path
from dulwich import porcelain
from dulwich.repo import Repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 70)
    print("ACADEMIC INTEGRITY RESEARCH PROJECT - GITHUB PUBLISHER")
    print("Target Repository: https://github.com/chetan926/ml.git")
    print("=" * 70)

    repo = Repo(str(PROJECT_ROOT))
    head_sha = repo.head().decode("ascii")
    print(f"\nLocal HEAD Commit: {head_sha}")
    print("Branch to push:    refs/heads/main")

    # Check environment variable first
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        if len(sys.argv) > 1:
            token = sys.argv[1].strip()
        else:
            print("\nEnter your GitHub Personal Access Token (PAT).")
            print("Note: The token will not be displayed on the screen as you type/paste.")
            try:
                token = getpass.getpass("GitHub Token: ").strip()
            except Exception:
                token = input("GitHub Token: ").strip()

    if not token:
        print("\nERROR: No token provided. Exiting.")
        sys.exit(1)

    remote_url = f"https://chetan926:{token}@github.com/chetan926/ml.git"

    print("\nConnecting to GitHub and pushing 'main' branch...")
    try:
        result = porcelain.push(
            repo,
            remote_location=remote_url,
            refspecs="refs/heads/main",
        )
        print("\n" + "=" * 70)
        print("SUCCESS! PROJECT FULLY PUBLISHED TO GITHUB:")
        print("  https://github.com/chetan926/ml")
        print(f"  Pushed commit: {head_sha}")
        print("=" * 70)
    except Exception as e:
        print(f"\nPush failed with error: {type(e).__name__}: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure your token has 'repo' (Full control of private repositories) permissions.")
        print("2. For fine-grained tokens, grant 'Contents: Read and Write' on 'chetan926/ml'.")
        print("3. Verify that your token has not expired.")
        sys.exit(1)

if __name__ == "__main__":
    main()
