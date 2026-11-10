"""The competition protocol (rules.json): loading, and the README section generated from it.
Checking submissions and edits against it is the subnet project's job.

python -m wsp.rules doc     regenerate the protocol section of README.md from rules.json
"""
import re
import sys

from .common import REPO, read_json

RULES = read_json(REPO / "rules.json")


def doc():
    s, e, u = RULES["submission"], RULES["edit"], RULES["upload"]
    fmt = {"max_payload_bytes": lambda v: f"{v // 1024} KiB"}
    limits = ", ".join(f"{k.removeprefix('max_').replace('_', ' ')} <= {fmt.get(k, lambda v: f'{v:.0%}')(v)}"
                       for k, v in e["limits"].items())

    def files(sec):
        return "\n".join(f"- `{k}`{' (required)' if k in sec['required'] else ''}: {v}" for k, v in sec["files"].items())

    return "\n".join([
        f"Generated from `rules.json` (`python -m wsp.rules doc`); a competition pins this repository's commit and the "
        f"file's sha256. {RULES['about']}", "",
        f"**Submission** `{s['path']}`. {s['about']}", files(s).replace("<entry_schema>", f'"{s["entry_schema"]}"'),
        f"- file types {', '.join(s['allowed_extensions'])}; at most {s['max_total_bytes'] // 1024} KiB in total", "",
        f"**Edit.** {e['about']}", f"- inheritance: {e['inheritance']}",
        f"- editable tensors: {e['editable_tensors']}", f"- ops: {', '.join(f'`{o}`' for o in e['ops'])}",
        f"- payload: {e['payload']}", f"- relative change: {e['relative_change']}", f"- **limits**: {limits}", "",
        f"**Upload** (`{u['schema']}`). {u['about']}", files(u), f"- {u['manifest'].replace('<schema>', u['schema'])}"])


def main():
    if sys.argv[1:] != ["doc"]:
        raise SystemExit(__doc__)
    readme = REPO / "README.md"
    block = f"<!-- protocol:begin -->\n{doc()}\n<!-- protocol:end -->"
    readme.write_text(re.sub(r"<!-- (rules|protocol):begin -->.*<!-- (rules|protocol):end -->", lambda _: block,
                             readme.read_text(), flags=re.S))


if __name__ == "__main__":
    main()
