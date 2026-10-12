#!/usr/bin/env python3
"""Read-only Markdown checks; never execute examples or contact a server."""
from __future__ import annotations

import argparse
import ast
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote


FENCES = re.compile(r"^```([^\n]*)\n(.*?)^```[ \t]*$", re.M | re.S)
LINKS = re.compile(r"\[[^\]\n]+\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[\"'][^\n]*?[\"'])?\)")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PRIVATE_NETWORKS = tuple(ipaddress.ip_network(value) for value in
                         ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
PERSONAL_PATH = re.compile(r"/(?:home|Users)/[^\s/<>`]+|[A-Za-z]:[\\/]Users[\\/][^\s<>`]+")


def markdown_files(root: Path) -> list[Path]:
    """Honor Git exclusions, including unpublished captures and upstream trees."""
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True, timeout=10)
        if Path(top.stdout.strip()).resolve() == root:
            listing = subprocess.run(
                ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                capture_output=True, check=True, timeout=10)
            names = set(listing.stdout.decode("utf-8").split("\0"))
            return sorted(root / name for name in names if name.endswith(".md") and (root / name).is_file())
    except (OSError, subprocess.SubprocessError, UnicodeError):
        pass
    files = [root / "README.md", root / "AGENTS.md", root / "CHANGELOG.md"]
    for folder in ("docs", "skills"):
        files.extend((root / folder).rglob("*.md"))
    return sorted({path for path in files if path.is_file()})


def heading_anchors(content: str) -> set[str]:
    result, seen = set(), {}
    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", FENCES.sub("", content), re.M):
        label = LINKS.sub(lambda item: item[0].split("](", 1)[0][1:], match[1])
        label = re.sub(r"[`*~]", "", label).lower()
        slug = "".join(char for char in label if char.isalnum() or char in "-_ ").replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        result.add(slug + (f"-{count}" if count else ""))
    return result


def check(root: Path) -> dict:
    errors, skipped = [], []
    counts = {"markdown_files": 0, "local_links": 0, "python_blocks": 0,
              "json_blocks": 0, "shell_blocks": 0}
    bash = shutil.which("bash")

    def error(path, line, kind, detail):
        errors.append({"file": str(path.relative_to(root)), "line": line,
                       "kind": kind, "detail": detail})

    def python_block(code, path, line):
        counts["python_blocks"] += 1
        try:
            ast.parse(code)
        except SyntaxError as exc:
            error(path, line, "python_syntax", str(exc))

    for path in markdown_files(root):
        if not path.resolve().is_relative_to(root):
            error(path, 1, "external_markdown_symlink", "Markdown resolves outside this repository")
            continue
        counts["markdown_files"] += 1
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            error(path, 1, "read_error", str(exc))
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if PERSONAL_PATH.search(line):
                error(path, number, "personal_path", "Review a user-specific absolute path")
            for candidate in IPV4.findall(line):
                try:
                    address = ipaddress.ip_address(candidate)
                except ValueError:
                    continue
                if any(address in network for network in PRIVATE_NETWORKS):
                    error(path, number, "private_address", "Review an RFC1918 address before publishing")

        # Preserve line numbers while excluding fenced examples from link checks.
        prose = FENCES.sub(lambda match: "\n" * match[0].count("\n"), content)
        for match in LINKS.finditer(prose):
            target = match[1] or match[2]
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target) or target.startswith("//"):
                continue
            counts["local_links"] += 1
            line = prose[:match.start()].count("\n") + 1
            location, _, anchor = target.partition("#")
            resolved = (path.parent / unquote(location)).resolve() if location else path.resolve()
            if not resolved.is_relative_to(root):
                error(path, line, "external_local_link", target)
            elif not resolved.exists():
                error(path, line, "missing_link", target)
            elif anchor and resolved.suffix == ".md":
                try:
                    if unquote(anchor) not in heading_anchors(resolved.read_text(encoding="utf-8")):
                        error(path, line, "missing_anchor", target)
                except (OSError, UnicodeError) as exc:
                    error(path, line, "read_error", str(exc))

        for block in FENCES.finditer(content):
            language, code = block[1].strip(), block[2]
            line = content[:block.start()].count("\n") + 2
            if language in ("python", "py"):
                python_block(code, path, line)
            elif language == "json":
                counts["json_blocks"] += 1
                try:
                    json.loads(code)
                except ValueError as exc:
                    error(path, line, "json_syntax", str(exc))
            elif language in ("bash", "sh", "shell"):
                counts["shell_blocks"] += 1
                if bash:
                    try:
                        result = subprocess.run([bash, "--noprofile", "--norc", "-n"], input=code,
                                                capture_output=True, text=True, timeout=10,
                                                env={"PATH": os.defpath, "LC_ALL": "C"})
                        if result.returncode:
                            error(path, line, "shell_syntax", result.stderr.strip())
                    except (OSError, subprocess.SubprocessError) as exc:
                        error(path, line, "shell_check_error", str(exc))
                else:
                    skipped.append({"file": str(path.relative_to(root)), "line": line,
                                    "reason": "bash unavailable; shell syntax not checked"})
                for heredoc in re.finditer(r"python(?:3)?\s+-\s+<<'?(\w+)'?\n(.*?)^\1$", code, re.M | re.S):
                    python_block(heredoc[2], path, line + code[:heredoc.start(2)].count("\n"))
    return {"status": "failed" if errors else "passed_with_skips" if skipped else "passed",
            "counts": counts, "errors": errors, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3],
                        help="Repository root; defaults to this skill's containing repository")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "src/musaprobe").is_dir():
        parser.error("--root must contain pyproject.toml and src/musaprobe")
    result = check(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
