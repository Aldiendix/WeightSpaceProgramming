"""Upload a finished task's deliverables to R2 by the fixed protocol in rules.json -> upload (mms.upload/1).

python -m wsp.upload <case> <edit> [--report FILE] [--task task.json] [--dry-run]

1. package under $WSP_WORK/deliver/<case>__<edit>/: cases/<case>/edits/<edit>.json -> edit/edit.json (base, parent,
   ops; a parent is referenced as parent/edit.json, not uploaded: it is already published) with its arrays in
   edit/edit.npz, and the report -> edit/REPORT.md (default: the case's NOTES.md + RESULTS.md);
2. upload every file to <prefix>/<path>, then manifest.json last; then read the manifest back and HEAD every file.
Credentials: --task (the subnet's task.json, section "r2": endpoint, bucket, prefix, access_key_id,
secret_access_key, session_token) or R2_* environment variables.
"""
import argparse
import hashlib
import os
import json
import shutil
from pathlib import Path

from .common import CASES, WORK, read_json, write_json
from .edits import export
from .rules import RULES


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package(case, edit, report=None):
    out = WORK / "deliver" / f"{case}__{edit}"
    shutil.rmtree(out, ignore_errors=True)
    export(CASES / case / "edits" / f"{edit}.json", out / "edit", "edit", parents=False)
    parts = [Path(report)] if report else [p for p in (CASES / case / "NOTES.md", CASES / case / "RESULTS.md") if p.exists()]
    if not parts:
        raise SystemExit("no report: pass --report or write the case's NOTES.md")
    (out / "edit" / "REPORT.md").write_text("\n\n".join(p.read_text() for p in parts))
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {"schema": RULES["upload"]["schema"],
                "files": [{"path": p.relative_to(out).as_posix(), "size": p.stat().st_size, "sha256": _sha(p)}
                          for p in files]}
    write_json(out / "manifest.json", manifest)
    return out, manifest


def target(task=None):
    """(endpoint, bucket, prefix, credentials) from the subnet's task.json, or from R2_* environment variables."""
    if task:
        r = read_json(task)["r2"]
        return r["endpoint"], r["bucket"], r["prefix"], (r["access_key_id"], r["secret_access_key"], r.get("session_token"))
    env = os.environ
    if not env.get("R2_BUCKET"):
        raise SystemExit("no R2 target: pass --task task.json or set R2_ENDPOINT, R2_BUCKET, R2_PREFIX, "
                         "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (and R2_SESSION_TOKEN for temporary keys)")
    return env["R2_ENDPOINT"], env["R2_BUCKET"], env.get("R2_PREFIX", ""), \
        (env["R2_ACCESS_KEY_ID"], env["R2_SECRET_ACCESS_KEY"], env.get("R2_SESSION_TOKEN") or None)


def upload(out, manifest, endpoint, bucket, prefix, creds, client=None):
    if client is None:
        import boto3  # noqa: PLC0415
        client = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=creds[0],
                              aws_secret_access_key=creds[1], aws_session_token=creds[2])

    def key(rel):
        return "/".join(x.strip("/") for x in (prefix, rel) if x.strip("/"))

    for f in manifest["files"]:  # files first ...
        client.upload_file(str(out / f["path"]), bucket, key(f["path"]))
    client.upload_file(str(out / "manifest.json"), bucket, key("manifest.json"))  # ... manifest last
    remote = json.loads(client.get_object(Bucket=bucket, Key=key("manifest.json"))["Body"].read())
    bad = [f["path"] for f in remote["files"]
           if client.head_object(Bucket=bucket, Key=key(f["path"]))["ContentLength"] != f["size"]]
    if remote != manifest or bad:
        raise SystemExit(f"upload incomplete: manifest {'ok' if remote == manifest else 'differs'}, bad files {bad}")
    return [key(f["path"]) for f in manifest["files"]] + [key("manifest.json")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--report", help="report file (default: the case's NOTES.md + RESULTS.md)")
    ap.add_argument("--task", help="the subnet's task.json (R2 target and temporary credentials)")
    ap.add_argument("--dry-run", action="store_true", help="package and print the manifest only")
    a = ap.parse_args()
    out, manifest = package(a.case, a.edit, a.report)
    print(json.dumps(manifest, indent=1))
    if not a.dry_run:
        print("\n".join(upload(out, manifest, *target(a.task))))


if __name__ == "__main__":
    main()
