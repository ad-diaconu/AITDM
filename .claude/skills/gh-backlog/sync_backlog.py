"""
One-way sync BACKLOG.md -> GitHub issues + the "AITDM project planning" project (v2, #2).

BACKLOG.md (English, repo root) is the single source of truth for which tasks exist and for their title,
description, branch, priority and status. Every row becomes one issue "<ID>: <title>", identified by the marker
`<!-- backlog-id: ID -->` in its body (or, for issues created before the marker existed, by the "<ID>: " title
prefix), added to the project with Status and Priority set. Re-running is idempotent. Nothing is deleted.

  status in BACKLOG.md   -> project Status / issue state
  todo, blocked          -> Backlog      / open   (blocked also gets the `blocked` label)
  ready                  -> Ready        / open   (planned and reviewed, can start)
  in progress            -> In progress  / open
  in review              -> In review    / open   (PR open)
  done                   -> Done         / closed (completed)
  rejected               -> Done         / closed (not planned), label `rejected`

An issue that is closed on GitHub (usually by a merged PR with "Fixes #N") while BACKLOG.md still says
todo / in progress is NOT reopened: the script warns, and you mark the row done.

Usage (from the repo root; the default is a dry run that only prints the plan):
  python .claude/skills/gh-backlog/sync_backlog.py            # dry run
  python .claude/skills/gh-backlog/sync_backlog.py --apply    # create / update
  python .claude/skills/gh-backlog/sync_backlog.py --apply --only E1 B2
Standard library only: any Python 3.9+ works (e.g. .venv/Scripts/python.exe on Windows).

IDs: E<n> = experiment (label `experiment`), B<n> = build step of the pipeline (label `build`).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

REPO = "ad-diaconu/AITDM"
OWNER = "ad-diaconu"
PROJECT_NUMBER = 2
BRANCH = "main"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BACKLOG = os.path.join(ROOT, "BACKLOG.md")
MARKER = "<!-- backlog-id: {} -->"

SECTION_LABELS = {"E": "experiment", "B": "build"}
LABEL_COLORS = {"experiment": "1d76db", "build": "c5def5", "rejected": "b60205", "blocked": "e99695"}
# keys are matched as prefixes of the status cell, so "in progress" / "in review" must come before shorter keys
STATUS_MAP = {"todo": ("Backlog", "open"), "blocked": ("Backlog", "open"), "ready": ("Ready", "open"),
              "in progress": ("In progress", "open"), "in review": ("In review", "open"),
              "done": ("Done", "completed"), "rejected": ("Done", "not planned")}
EXTRA_LABEL = {"blocked": "blocked", "rejected": "rejected"}
# | ID | Title | Description | Branch | P | Status |
ROW = re.compile(r"^\| ([EB]\d+) \| (.+?) \| (.*?) \| (.*?) \| (P\d) \| (.*?) \|\s*$")


def gh(*args, as_json=False):
    r = subprocess.run(["gh", *args], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}\n{r.stderr.strip()}")
    return json.loads(r.stdout) if as_json else r.stdout.strip()


def parse_backlog(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = ROW.match(line.rstrip("\n"))
            if not m:
                continue
            bid, title, desc, branch, prio, status_raw = m.groups()
            status = next((s for s in STATUS_MAP if status_raw.strip().lower().startswith(s)), None)
            if status is None:
                print(f"WARNING: {bid}: unknown status {status_raw!r}, treated as todo")
                status = "todo"
            items.append({"id": bid, "title": title.strip(), "desc": desc.strip(), "branch": branch.strip(),
                          "priority": prio, "status": status})
    return items


def absolutize(md):
    """Relative repo links -> GitHub blob URLs, so they work inside the issue."""
    def fix(m):
        target = m.group(2)
        if re.match(r"^(https?:|#|mailto:)", target):
            return m.group(0)
        return f"[{m.group(1)}](https://github.com/{REPO}/blob/{BRANCH}/{target.lstrip('./')})"
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix, md)


def build(it):
    labels = [SECTION_LABELS[it["id"][0]]]
    if it["status"] in EXTRA_LABEL:
        labels.append(EXTRA_LABEL[it["status"]])
    body = absolutize(it["desc"])
    if it["branch"] and it["branch"] not in ("-", "—"):
        body += f"\n\nBranch: {it['branch']}"
    body += f"\n\n<sub>Synced from BACKLOG.md. Edit there, not here.</sub>\n{MARKER.format(it['id'])}\n"
    return f"{it['id']}: {it['title']}", body, labels


def project_meta():
    proj = gh("project", "view", str(PROJECT_NUMBER), "--owner", OWNER, "--format", "json", as_json=True)
    fields = gh("project", "field-list", str(PROJECT_NUMBER), "--owner", OWNER, "--format", "json",
                as_json=True)["fields"]
    opts = {f["name"]: {"id": f["id"], "options": {o["name"]: o["id"] for o in f.get("options", [])}}
            for f in fields if f["name"] in ("Status", "Priority")}
    items = gh("project", "item-list", str(PROJECT_NUMBER), "--owner", OWNER, "--format", "json",
               "--limit", "1000", as_json=True)["items"]
    by_issue = {i["content"]["number"]: i["id"] for i in items if i.get("content", {}).get("number")}
    return proj["id"], opts, by_issue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    items = [i for i in parse_backlog(BACKLOG) if not a.only or i["id"] in a.only]
    issues = gh("issue", "list", "--repo", REPO, "--state", "all", "--limit", "1000",
                "--json", "number,title,body,state,labels,url", as_json=True)
    by_id = {}
    for iss in issues:
        m = re.search(r"<!-- backlog-id: ([EB]\d+) -->", iss.get("body") or "")
        if m and m.group(1) not in by_id:
            by_id[m.group(1)] = iss
        elif m:  # two issues carry the same id: keep the older one, warn
            older = min(by_id[m.group(1)], iss, key=lambda x: x["number"])
            print(f"WARNING: issues #{by_id[m.group(1)]['number']} and #{iss['number']} both carry {m.group(1)}; "
                  f"using #{older['number']}")
            by_id[m.group(1)] = older
    for iss in issues:  # issues created before the marker: match by the "<ID>: " title prefix
        m = re.match(r"^\[?([EB]\d+)\]?:? ", iss["title"])
        if m and m.group(1) not in by_id:
            by_id[m.group(1)] = iss
    print(f"{len(items)} backlog items to sync, {len(by_id)} already mirrored as issues")

    if a.apply:
        existing = {label["name"] for label in gh("label", "list", "--repo", REPO, "--limit", "200", "--json", "name",
                                          as_json=True)}
        for name, color in LABEL_COLORS.items():
            if name not in existing:
                gh("label", "create", name, "--repo", REPO, "--color", color)
        try:
            gh("project", "link", str(PROJECT_NUMBER), "--owner", OWNER, "--repo", REPO)
        except RuntimeError:
            pass  # already linked
        project_id, fields, proj_items = project_meta()

    for it in items:
        title, body, labels = build(it)
        status_opt, state = STATUS_MAP[it["status"]]
        iss = by_id.get(it["id"])
        if iss is not None and iss["state"] == "CLOSED" and state == "open":
            print(f"  WARNING #{iss['number']} {title}: closed on GitHub but '{it['status']}' in BACKLOG.md; "
                  f"not reopened. Mark it done (or rejected) in BACKLOG.md")
            status_opt, state = "Done", None
        print(f"  {'create' if iss is None else 'update':6} {title}  | {it['priority']} | {status_opt} | {', '.join(labels)}")
        if not a.apply:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8", newline="\n") as f:
            f.write(body)
            body_file = f.name
        try:
            if iss is None:
                args = ["issue", "create", "--repo", REPO, "--title", title, "--body-file", body_file]
                for lab in labels:
                    args += ["--label", lab]
                url = gh(*args).splitlines()[-1]
                number, cur_state = int(url.rstrip("/").split("/")[-1]), "OPEN"
            else:
                number, url, cur_state = iss["number"], iss["url"], iss["state"]
                args = ["issue", "edit", str(number), "--repo", REPO, "--title", title, "--body-file", body_file]
                for lab in labels:
                    args += ["--add-label", lab]
                for lab in (label["name"] for label in iss["labels"]):
                    if lab in LABEL_COLORS and lab not in labels:
                        args += ["--remove-label", lab]
                gh(*args)
        finally:
            os.unlink(body_file)
        if state not in (None, "open") and cur_state == "OPEN":
            gh("issue", "close", str(number), "--repo", REPO, "--reason", state)
        item_id = proj_items.get(number)
        if item_id is None:
            try:
                item_id = gh("project", "item-add", str(PROJECT_NUMBER), "--owner", OWNER, "--url", url,
                             "--format", "json", as_json=True)["id"]
            except RuntimeError as e:
                if "already exists" not in str(e) and "conflict" not in str(e):
                    raise
                # the project's auto-add workflow already added the new issue (with a short delay)
                for _ in range(10):
                    proj_items = project_meta()[2]
                    if number in proj_items:
                        break
                    time.sleep(3)
                item_id = proj_items[number]
        for field, value in (("Status", status_opt), ("Priority", it["priority"])):
            gh("project", "item-edit", "--id", item_id, "--project-id", project_id,
               "--field-id", fields[field]["id"], "--single-select-option-id", fields[field]["options"][value])
    if not a.apply:
        print("\n(dry run — nothing changed; re-run with --apply)")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
