import json
import re

import numpy as np

from wsp import upload
from wsp.rules import RULES, doc


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[key] = open(path, "rb").read()

    def get_object(self, Bucket, Key):  # noqa: N803
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": len(self.objects[Key])}


def test_package_and_upload(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    np.savez(case / "edits" / "e.npz", u15=np.ones(4, np.float32), cols15=np.arange(3))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "meta": {"directions": "x.npz"}, "ops": [
        {"op": "dir_ablate", "tensor": "t", "u": "e.npz#u15", "cols": "e.npz#cols15", "alpha": 6}]}))
    (case / "NOTES.md").write_text("notes")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    assert {f["path"] for f in man["files"]} == {"edit/edit.json", "edit/edit.npz", "edit/REPORT.md"}
    spec = json.loads((out / "edit" / "edit.json").read_text())
    assert spec["ops"][0]["u"] == "edit.npz#u15" and "meta" not in spec
    s3 = FakeS3()
    keys = upload.upload(out, man, "e", "bucket", "staging/sid/", (None, None, None), client=s3)
    assert keys[-1] == "staging/sid/manifest.json" and list(s3.objects)[-1] == keys[-1]  # manifest last
    assert json.loads(s3.objects["staging/sid/manifest.json"])["schema"] == RULES["upload"]["schema"]


def test_package_includes_parent_chain(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    (case / "parent").mkdir()
    (case / "parent" / "parent.json").write_text(json.dumps({"name": "parent", "base": "b", "ops": []}))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "parent": "../parent/parent.json",
                                                       "ops": []}))
    (case / "NOTES.md").write_text("n")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    assert "edit/parent/parent.json" in {f["path"] for f in man["files"]}
    assert json.loads((out / "edit" / "edit.json").read_text())["parent"] == "parent/parent.json"


def test_published_parent_is_copied_verbatim(tmp_path, monkeypatch):
    pub = tmp_path / "published" / "edit"  # a published submission's edit/ directory
    (pub / "derive").mkdir(parents=True)
    np.savez(pub / "edit.npz", u15=np.ones(4, np.float32))
    (pub / "edit.json").write_text(json.dumps({"name": "edit", "base": "b", "ops": [
        {"op": "dir_ablate", "tensor": "t", "u": "edit.npz#u15"}]}))
    (pub / "REPORT.md").write_text("r")
    (pub / "derive" / "d.py").write_text("x = 1")
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    from wsp.edits import copy_parent
    copy_parent(pub / "edit.json", case / "parent")
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "parent": "../parent/edit.json",
                                                       "ops": []}))
    (case / "NOTES.md").write_text("n")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    sha = {f["path"]: f["sha256"] for f in man["files"]}
    for p in pub.rglob("*"):
        if p.is_file():
            assert sha["edit/parent/" + p.relative_to(pub).as_posix()] == upload._sha(p)  # byte-identical
    assert json.loads((out / "edit" / "edit.json").read_text())["parent"] == "parent/edit.json"


def test_protocol():
    pat = RULES["submission"]["path_pattern"]
    assert re.match(pat, "submissions/5" + "A" * 47 + "/my-entry/")
    assert not re.match(pat, "submissions/team/my-entry/")
    assert set(RULES["submission"]["required"]) <= set(RULES["submission"]["files"])
    assert set(RULES["upload"]["required"]) <= set(RULES["upload"]["files"])
    text = doc()
    assert "payload" in text and RULES["upload"]["schema"] in text
