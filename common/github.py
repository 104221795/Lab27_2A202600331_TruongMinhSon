"""Thin wrapper around the GitHub REST API.

We talk to GitHub directly via `httpx` instead of shelling out to `gh`, so
students don't need a `gh` binary — just `GITHUB_TOKEN` in `.env`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import httpx


API = "https://api.github.com"
PR_URL_RE = re.compile(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)")


@dataclass
class PullRequest:
    url: str
    owner: str
    repo: str
    number: int
    title: str
    author: str
    base_ref: str
    head_ref: str
    head_sha: str
    diff: str
    files_changed: list[str]


def _token() -> str:
    tok = os.environ.get("GITHUB_TOKEN")
    if not tok:
        raise RuntimeError(
            "GITHUB_TOKEN is not set. Copy .env.example to .env and paste a Personal "
            "Access Token (https://github.com/settings/tokens/new) with `public_repo` scope."
        )
    return tok


def _headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Day27-HITL-Lab",
    }


def parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    m = PR_URL_RE.search(pr_url)
    if not m:
        raise ValueError(f"Not a PR URL: {pr_url}")
    return m.group(1), m.group(2), int(m.group(3))


def fetch_pr(pr_url: str) -> PullRequest:
    """Fetch PR metadata + unified diff via the GitHub REST API."""
    owner, repo, number = parse_pr_url(pr_url)
    base = f"{API}/repos/{owner}/{repo}/pulls/{number}"

    with httpx.Client(timeout=30.0) as client:
        meta_resp = client.get(base, headers=_headers())
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        diff_resp = client.get(
            base, headers=_headers(accept="application/vnd.github.v3.diff")
        )
        diff_resp.raise_for_status()
        diff = diff_resp.text

        files_resp = client.get(f"{base}/files", headers=_headers())
        files_resp.raise_for_status()
        files = [f["filename"] for f in files_resp.json()]

    return PullRequest(
        url=pr_url,
        owner=owner,
        repo=repo,
        number=number,
        title=meta["title"],
        author=meta["user"]["login"],
        base_ref=meta["base"]["ref"],
        head_ref=meta["head"]["ref"],
        head_sha=meta["head"]["sha"],
        diff=diff,
        files_changed=files,
    )


def post_review_comment(pr_url: str, body: str) -> str | None:
    """Post a top-level discussion comment back to the PR.

    Uses the Issues endpoint (PRs are issues under the hood for top-level
    comments). For formal Approve/Request-changes use the Reviews endpoint
    instead — which requires collaborator status on the target repo.

    Takes the PR URL directly so callers don't need a `PullRequest` object.
    """
    owner, repo, number = parse_pr_url(pr_url)
    url = f"{API}/repos/{owner}/{repo}/issues/{number}/comments"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, headers=_headers(), json={"body": body})
        resp.raise_for_status()
        return resp.json().get("html_url")
