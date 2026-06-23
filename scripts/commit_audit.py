#!/usr/bin/env python3
"""Find commits authored on weekdays during work hours.

Scans every reachable commit across all refs (local branches, remotes, tags)
and flags those whose author timestamp falls on a weekday between a
configurable start and end hour (default 08:00 inclusive, 17:00 exclusive)
in the author's local timezone.

By default filters to the current git user.email. Use --all-authors to
include everyone, or --author EMAIL to pin a specific author.

Examples:
    scripts/commit_audit.py
    scripts/commit_audit.py --start 9 --end 18
    scripts/commit_audit.py --author edevlin64@gmail.com
    scripts/commit_audit.py --all-authors
    scripts/commit_audit.py --diff a4e5f65
    scripts/commit_audit.py --diff a4e5f65 --stat-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Commit:
    sha: str
    author_email: str
    author_iso: str
    subject: str

    @property
    def dt(self) -> datetime:
        return datetime.fromisoformat(self.author_iso)


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def fetch_commits() -> list[Commit]:
    """Pull every reachable commit across all refs, deduped by SHA."""
    raw = run_git([
        "log",
        "--all",
        "--reflog",
        "--pretty=format:%H%x09%aE%x09%aI%x09%s",
    ])
    seen: set[str] = set()
    commits: list[Commit] = []
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split("\t", 3)
        if len(parts) != 4:
            continue
        sha, email, iso, subject = parts
        if sha in seen:
            continue
        seen.add(sha)
        commits.append(Commit(sha, email, iso, subject))
    return commits


def branches_containing(sha: str) -> list[str]:
    """Return refs containing this commit, abbreviated for display."""
    out = run_git(["branch", "--all", "--contains", sha])
    refs = []
    for line in out.splitlines():
        ref = line.strip().lstrip("* ").strip()
        if not ref or ref.startswith("(HEAD"):
            continue
        refs.append(ref)
    return refs


def in_work_window(c: Commit, start_hour: int, end_hour: int) -> bool:
    dt = c.dt
    if dt.weekday() >= 5:  # 5=Sat, 6=Sun
        return False
    return start_hour <= dt.hour < end_hour


def cmd_list(args: argparse.Namespace) -> int:
    commits = fetch_commits()

    target_emails: set[str] | None = None
    if not args.all_authors:
        email = args.author or run_git(["config", "user.email"]).strip()
        target_emails = {email}

    flagged = []
    for c in commits:
        if target_emails and c.author_email not in target_emails:
            continue
        if in_work_window(c, args.start, args.end):
            flagged.append(c)

    flagged.sort(key=lambda c: c.dt)

    window = f"weekdays {args.start:02d}:00-{args.end:02d}:00 (local)"
    scope = "all authors" if args.all_authors else next(iter(target_emails or {"<unknown>"}))

    if not flagged:
        print(f"No commits found in {window} for {scope}. Total commits scanned: {len(commits)}.")
        return 0

    print(f"Found {len(flagged)} commits in {window} for {scope} (of {len(commits)} total):")
    print()
    header = f"{'SHA':<10}  {'DAY':<3}  {'DATETIME':<26}  {'AUTHOR':<28}  SUBJECT"
    print(header)
    print("-" * len(header))
    for c in flagged:
        wd = c.dt.strftime("%a")
        ts = c.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        subject = c.subject if len(c.subject) <= 80 else c.subject[:77] + "..."
        print(f"{c.sha[:9]:<10}  {wd:<3}  {ts:<26}  {c.author_email:<28}  {subject}")

    if args.show_refs:
        print()
        print("Refs containing each flagged commit:")
        for c in flagged:
            refs = branches_containing(c.sha)
            joined = ", ".join(refs) if refs else "(none)"
            print(f"  {c.sha[:9]}  {joined}")

    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    sha = args.diff
    try:
        run_git(["cat-file", "-e", f"{sha}^{{commit}}"])
    except subprocess.CalledProcessError:
        print(f"error: commit {sha!r} not found", file=sys.stderr)
        return 1
    log_args = ["show", "--stat"]
    if not args.stat_only:
        log_args.append("--patch")
    log_args.append(sha)
    sys.stdout.write(run_git(log_args))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", type=int, default=8,
                   help="Window start hour, inclusive (24h). Default 8.")
    p.add_argument("--end", type=int, default=17,
                   help="Window end hour, exclusive (24h). Default 17.")
    p.add_argument("--author", metavar="EMAIL",
                   help="Filter by this author email. Defaults to git config user.email.")
    p.add_argument("--all-authors", action="store_true",
                   help="Include commits from all authors.")
    p.add_argument("--show-refs", action="store_true",
                   help="After the list, print which refs contain each flagged commit.")
    p.add_argument("--diff", metavar="SHA",
                   help="Show the diff for a specific commit instead of listing.")
    p.add_argument("--stat-only", action="store_true",
                   help="With --diff, show only the diffstat (no patch body).")
    args = p.parse_args()

    if not (0 <= args.start < 24 and 0 < args.end <= 24 and args.start < args.end):
        print("error: --start and --end must satisfy 0 <= start < end <= 24",
              file=sys.stderr)
        return 2

    if args.diff:
        return cmd_diff(args)
    return cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
