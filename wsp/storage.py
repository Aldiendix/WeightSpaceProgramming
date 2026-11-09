"""Upload task outputs to the temporary R2 directory (S3-compatible; R2_* settings in .config).
Keys: <R2_PREFIX>/<relative key>. Disabled when R2_BUCKET is empty.

python -m wsp.storage upload <path>... [--key <relative key or dir>]   (directories recursively)
python -m wsp.storage ls [<relative prefix>]
"""
from pathlib import Path

from . import config


def enabled():
    return bool(config.get("R2_BUCKET"))


def _client():
    import boto3  # noqa: PLC0415
    return boto3.client("s3", endpoint_url=config.get("R2_ENDPOINT"), region_name="auto",
                        aws_access_key_id=config.get("R2_ACCESS_KEY_ID"),
                        aws_secret_access_key=config.get("R2_SECRET_ACCESS_KEY"))


def _key(rel):
    return "/".join(p.strip("/") for p in (config.get("R2_PREFIX"), str(rel)) if p.strip("/"))


def upload(path, rel, client=None):
    """Upload a file (or every file under a directory) to <prefix>/<rel>; returns the uploaded keys."""
    client, path = client or _client(), Path(path)
    files = [(path, rel)] if path.is_file() else [(p, f"{rel}/{p.relative_to(path).as_posix()}")
                                                   for p in sorted(path.rglob("*")) if p.is_file()]
    keys = []
    for p, r in files:
        client.upload_file(str(p), config.get("R2_BUCKET"), _key(r))
        keys.append(_key(r))
    return keys


def upload_if_enabled(items):
    """items: [(local path, relative key)]. No-op without R2 settings."""
    if not enabled():
        return []
    client = _client()
    return [k for p, rel in items if Path(p).exists() for k in upload(p, rel, client)]


def main():
    import argparse  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("paths", nargs="+")
    u.add_argument("--key", help="relative key (single path) or key prefix (several paths); default: the path's name")
    ls = sub.add_parser("ls")
    ls.add_argument("prefix", nargs="?", default="")
    a = ap.parse_args()
    if not enabled():
        raise SystemExit("R2 is not configured (R2_BUCKET in .config)")
    if a.cmd == "upload":
        for p in a.paths:
            rel = a.key if a.key and len(a.paths) == 1 else "/".join(x for x in (a.key, Path(p).name) if x)
            print("\n".join(upload(p, rel)))
    else:
        pages = _client().get_paginator("list_objects_v2").paginate(Bucket=config.get("R2_BUCKET"), Prefix=_key(a.prefix))
        for page in pages:
            for o in page.get("Contents", []):
                print(f"{o['Size']:>12}  {o['Key']}")


if __name__ == "__main__":
    main()
