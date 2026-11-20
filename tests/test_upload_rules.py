import json
import os
import re

import numpy as np

from wsp import case as wsp_case
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
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    assert {f["path"] for f in man["files"]} == {"edit/edit.json", "edit/edit.npz", "edit/REPORT.md"}
    spec = json.loads((out / "edit" / "edit.json").read_text())
    assert spec["ops"][0]["u"] == "edit.npz#u15" and "meta" not in spec
    s3 = FakeS3()
    keys = upload.upload(out, man, "e", "bucket", "staging/sid/", (None, None, None), client=s3)
    assert keys[-1] == "staging/sid/manifest.json" and list(s3.objects)[-1] == keys[-1]  # manifest last
    assert json.loads(s3.objects["staging/sid/manifest.json"])["schema"] == RULES["upload"]["schema"]


def test_upload_references_parent_without_copying_it(tmp_path, monkeypatch):
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    (case / "parent").mkdir()
    (case / "parent" / "edit.json").write_text(json.dumps({"base": "b", "ops": []}))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "parent": "../parent/edit.json",
                                                       "ops": []}))
    (case / "NOTES.md").write_text("n")
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    assert {f["path"] for f in man["files"]} == {"edit/edit.json", "edit/REPORT.md"}  # nothing else
    assert json.loads((out / "edit" / "edit.json").read_text()) == {"base": "b", "parent": "parent/edit.json", "ops": []}


def test_upload_references_the_start_by_its_task_directory(tmp_path, monkeypatch):
    from wsp.edits import copy_parent, start_dir
    task = tmp_path / "task"
    for d in ("parent", "champion"):
        (task / d / "edit").mkdir(parents=True)
        (task / d / "edit" / "edit.json").write_text(json.dumps({"base": "b", "ops": []}))
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    for d in ("parent", "champion"):
        case = tmp_path / "cases" / d
        start = copy_parent(task / d / "edit" / "edit.json", case / start_dir(task / d / "edit" / "edit.json"))
        (case / "edits").mkdir()
        (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "ops": [],
                                                           "parent": os.path.relpath(start, case / "edits")}))
        (case / "NOTES.md").write_text("n")
        out, _ = upload.package(d, "e")
        assert json.loads((out / "edit" / "edit.json").read_text())["parent"] == f"{d}/edit.json"


def test_published_parent_is_copied_verbatim(tmp_path):
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
    for p in pub.rglob("*"):  # a case starts from a byte-identical copy of the published edit
        if p.is_file():
            assert upload._sha(case / "parent" / p.relative_to(pub)) == upload._sha(p)


def test_protocol():
    pat = RULES["submission"]["path_pattern"]
    assert re.match(pat, "submissions/5" + "A" * 47 + "/my-entry/")
    assert not re.match(pat, "submissions/team/my-entry/")
    assert set(RULES["submission"]["required"]) <= set(RULES["submission"]["files"])
    assert set(RULES["upload"]["required"]) <= set(RULES["upload"]["files"])
    text = doc()
    assert "payload" in text and RULES["upload"]["schema"] in text


def test_report_carries_every_result_and_status_reads_the_upload_back(tmp_path, monkeypatch):
    """The packaged REPORT.md ends with the case's results table, where a holdout run made before its edit was frozen
    is marked; --status tells this package from another one."""
    case = tmp_path / "cases" / "c"
    (case / "edits").mkdir(parents=True)
    (case / "results").mkdir()
    (case / "case.json").write_text(json.dumps({"profile": "p", "contrast": "loop", "frozen": "e"}))
    (case / "edits" / "e.json").write_text(json.dumps({"name": "e", "base": "b", "ops": []}))
    (case / "NOTES.md").write_text("notes")
    paired = {"base_acc": 0.5, "edit_acc": 0.6, "gain": 0.1, "se": 0.02, "rescues": 3, "losses": 1, "base_cut": 0.2,
              "edit_cut": 0.1}
    for model, frozen in (("e", None), ("f", "e")):  # e ran on holdout before it was frozen; f is another edit
        (case / "results" / f"{model}__holdout256.json").write_text(json.dumps(
            {"summary": {}, "paired": paired, "frozen": frozen}))
    monkeypatch.setattr(upload, "CASES", tmp_path / "cases")
    monkeypatch.setattr(wsp_case, "CASES", tmp_path / "cases")
    monkeypatch.setattr(upload, "WORK", tmp_path / "work")
    out, man = upload.package("c", "e")
    report = (out / "edit" / "REPORT.md").read_text()
    assert report.startswith("notes") and report.count("holdout256 (not frozen)") == 2
    s3 = FakeS3()
    assert upload.status(man, "e", "bucket", "p", (None,) * 3, client=s3)[0] is None  # nothing uploaded yet
    upload.upload(out, man, "e", "bucket", "p", (None,) * 3, client=s3)
    assert "this package" in upload.status(man, "e", "bucket", "p", (None,) * 3, client=s3)[1]
    other = {**man, "files": man["files"][:1]}
    assert "another package" in upload.status(other, "e", "bucket", "p", (None,) * 3, client=s3)[1]
