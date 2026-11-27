"""Pascal VOC export and the AI scorecard."""

import io
import zipfile

from .test_api import _setup


def test_voc_export_and_scorecard(client):
    pid, label_id, image = _setup(client)
    iid = image["id"]
    anns = [
        {"id": "k1", "label_id": label_id, "kind": "box", "source": "ai", "score": 0.9,
         "geometry": {"x": 2, "y": 3, "w": 10, "h": 20}},
        {"id": "f1", "label_id": label_id, "kind": "polygon", "source": "ai_edited", "score": 0.8,
         "geometry": {"points": [10, 10, 30, 10, 30, 30]}},
        {"id": "h1", "label_id": label_id, "kind": "box", "geometry": {"x": 40, "y": 5, "w": 10, "h": 10}},
    ]
    assert client.put(f"/api/images/{iid}/annotations", json={"annotations": anns}).status_code == 200
    events = [{"kind": "annotation_committed", "image_id": iid, "annotation_id": "gone",
               "payload": {"source": "ai", "label_id": label_id}},
              {"kind": "annotation_deleted", "image_id": iid, "annotation_id": "gone", "payload": {"source": "ai"}},
              {"kind": "annotation_deleted", "image_id": iid, "annotation_id": "k1", "payload": {"source": "ai"}}]
    assert client.post(f"/api/projects/{pid}/events", json={"events": events}).status_code == 204

    card = client.get(f"/api/projects/{pid}/scorecard").json()
    t = card["total"]
    assert card["images"] == 1
    # k1 was deleted then restored (Undo): still counts as kept, not removed.
    assert (t["ai_kept"], t["ai_fixed"], t["ai_removed"], t["hand"]) == (1, 1, 1, 1)
    assert abs(t["precision"] - 2 / 3) < 1e-9 and abs(t["recall"] - 2 / 3) < 1e-9
    assert "cat" in card["per_label"]
    dl = client.get(f"/api/projects/{pid}/scorecard?download=true")
    assert "attachment" in dl.headers["content-disposition"]

    res = client.get(f"/api/projects/{pid}/export?format=voc")
    assert res.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(res.content))
    xmls = [n for n in zf.namelist() if n.startswith("Annotations/")]
    xml = zf.read(xmls[0]).decode()
    assert xml.count("<object>") == 3 and "<name>cat</name>" in xml
    assert "<xmin>3</xmin><ymin>4</ymin><xmax>12</xmax><ymax>23</ymax>" in xml
    assert any(n.startswith("JPEGImages/") for n in zf.namelist()) and "ImageSets/Main/default.txt" in zf.namelist()


class _FakeYolo:
    def detect(self, rgb, conf, iou):
        return [{"score": 0.5, "label": "cat", "class_id": 0, "box": {"x": 1, "y": 1, "w": 5, "h": 5}}], 2.0


def test_next_images_ranks_unsure_first(client, monkeypatch):
    from app.inference import registry
    from app.routes.ai import uncertainty

    assert uncertainty([]) == 1.0 and uncertainty([0.5]) == 1.0 and uncertainty([0.99, 0.01]) < 0.05
    pid, _, image = _setup(client)
    assert client.post(f"/api/projects/{pid}/next-images").status_code == 503  # no weights
    monkeypatch.setattr(registry, "get_yolo", lambda: _FakeYolo())
    res = client.post(f"/api/projects/{pid}/next-images?limit=3").json()
    assert res["ranked"] == 1 and res["images"][0]["image_id"] == image["id"]
    assert res["images"][0]["unsure"] == 1 and res["images"][0]["uncertainty"] == 1.0


def test_reset_project_keeps_images(client):
    pid, label_id, image = _setup(client)
    iid = image["id"]
    client.put(f"/api/images/{iid}/annotations", json={"annotations": [
        {"id": "reset1", "label_id": label_id, "kind": "box", "geometry": {"x": 1, "y": 1, "w": 5, "h": 5}}]})
    client.patch(f"/api/images/{iid}", json={"status": "done"})
    sess = client.post(f"/api/projects/{pid}/sessions", json={"mode": "ai"}).json()
    client.post(f"/api/projects/{pid}/events", json={"events": [
        {"session_id": sess["id"], "kind": "annotation_committed", "image_id": iid, "annotation_id": "reset1"}]})

    r = client.post(f"/api/projects/{pid}/reset", json={}).json()
    assert r == {"annotations_deleted": 1, "images_reset": 1, "events_deleted": 0, "labels_deleted": 0}
    assert client.get(f"/api/images/{iid}/annotations").json() == []
    assert client.get(f"/api/images/{iid}").json()["status"] == "new"
    assert client.get(f"/api/projects/{pid}/stats").json()["per_mode"]["ai"]["objects"] == 1  # stats kept

    r = client.post(f"/api/projects/{pid}/reset", json={"clear_stats": True, "remove_labels": True}).json()
    assert r["events_deleted"] == 1 and r["labels_deleted"] == 1
    assert client.get(f"/api/projects/{pid}/stats").json()["per_mode"]["ai"]["objects"] == 0
    assert client.get(f"/api/projects/{pid}/labels").json() == []
    assert client.get(f"/api/projects/{pid}/images").json()[0]["id"] == iid  # image still there
