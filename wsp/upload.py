"""Upload a finished task's deliverables to R2 by the fixed protocol in rules.json -> upload.

python -m wsp.upload <case> <edit> [--report FILE] [--task task.json] [--start SID|base] [--dry-run | --status]

1. package under $WSP_WORK/deliver/<case>__<edit>/: cases/<case>/edits/<edit>.json -> edit/edit.json (base, parent,
   ops) with every array an op reads in edit/edit.npz (none: no edit.npz; inline lists stay in edit.json). "parent"
   is always written: the sid of the published edit it stacks on, which is the name the case keeps it under (init
   --parent <sid>/edit/edit.json keeps a task's start as <sid>/edit/; otherwise --start <sid>), not uploaded since it
   is already published; or the base as <hf_repo>@<revision>. With --task the sid must be one of the task's starts.
   Local edits in between are carried: their ops go into the upload. --start base carries the whole chain instead
   (e.g. a start the task does not offer); the limits then count those ops too;
   and the report -> edit/REPORT.md: --report or the case's NOTES.md, then the case's results table (every run,
   paired, holdout runs made before the edit was frozen marked). The package is rebuilt on every call: change the
   case's records or pass --report, not the package. An organizer's check runs on the package's edit/ directory;
2. upload every file to <prefix>/<path>, then manifest.json last; then read the manifest back and HEAD every file.
--status reads the remote manifest back and says whether it is this package; nothing is uploaded.
Credentials: --task (the subnet's task.json, section "r2": endpoint, bucket, prefix, access_key_id,
secret_access_key, session_token) or R2_* environment variables.
"""
import argparse
import hashlib
import os
import json
import shutil
from pathlib import Path

from . import profiles
from .common import CASES, WORK, read_json, write_json
from .edits import export_upload
from .rules import RULES


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def package(case, edit, report=None, start=None, task=None):
    out = WORK / "deliver" / f"{case}__{edit}"
    shutil.rmtree(out, ignore_errors=True)
    t = read_json(task) if task else {}
    starts = [x["sid"] for x in t["starts"]] if "starts" in t else None
    _, on, carried = export_upload(CASES / case / "edits" / f"{edit}.json", out / "edit", start, starts)
    base = profiles.base_name(read_json(out / "edit" / "edit.json")["base"])
    if t.get("base_name") and base != t["base_name"]:
        raise SystemExit(f"this edit's base is {base}, the task's {t['base_name']}")
    print(f"[upload] {edit} stacks on {on}" + (f", carrying the ops of {', '.join(carried)}" if carried else ""),
          flush=True)
    from .case import case_tables  # noqa: PLC0415
    src = Path(report) if report else CASES / case / "NOTES.md"
    if not src.exists():
        raise SystemExit("no report: pass --report or write the case's NOTES.md")
    (out / "edit" / "REPORT.md").write_text(src.read_text().rstrip() + "\n\n" + case_tables(case))
    print(f"[upload] package {out / 'edit'} (REPORT.md: {src} + the case's results table)", flush=True)
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


def status(manifest, endpoint, bucket, prefix, creds, client=None):
    """The remote manifest, and whether it is this package's."""
    if client is None:
        import boto3  # noqa: PLC0415
        client = boto3.client("s3", endpoint_url=endpoint, region_name="auto", aws_access_key_id=creds[0],
                              aws_secret_access_key=creds[1], aws_session_token=creds[2])
    key = "/".join(x.strip("/") for x in (prefix, "manifest.json") if x.strip("/"))
    try:
        obj = client.get_object(Bucket=bucket, Key=key)
    except Exception as ex:  # noqa: BLE001 - nothing uploaded yet, or no access
        return None, f"no manifest at r2://{bucket}/{key} ({type(ex).__name__})"
    remote = json.loads(obj["Body"].read())
    when = obj.get("LastModified")
    same = remote == manifest
    return remote, (f"r2://{bucket}/{key} ({when}): " + ("this package" if same else "another package (differs from "
                                                                                       "this one)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("edit")
    ap.add_argument("--report", help="report text (default: the case's NOTES.md); the results table is appended")
    ap.add_argument("--task", help="the subnet's task.json (R2 target and temporary credentials)")
    ap.add_argument("--start", help="the sid of the published start, when the case keeps it under another name; "
                                    "base: carry the whole chain")
    ap.add_argument("--dry-run", action="store_true", help="package and print the manifest only")
    ap.add_argument("--status", action="store_true", help="compare the remote manifest with this package; no upload")
    a = ap.parse_args()
    out, manifest = package(a.case, a.edit, a.report, a.start, a.task)
    if a.status:
        print(f"[upload] {status(manifest, *target(a.task))[1]}")
        return
    print(json.dumps(manifest, indent=1))
    if not a.dry_run:
        endpoint, bucket, prefix, creds = target(a.task)
        keys = upload(out, manifest, endpoint, bucket, prefix, creds)
        print("\n".join(f"[upload] r2://{bucket}/{k}" for k in keys))
        print(f"[upload] verified: the remote manifest equals the local one and its {len(manifest['files'])} files "
              "have their sizes; the last complete upload before the deadline counts")


if __name__ == "__main__":
    main()
