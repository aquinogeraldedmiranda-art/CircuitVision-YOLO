"""SQLite reference database tests, with no YOLO accuracy assumptions."""
import io
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import component_db
from backend import main


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    path=tmp_path / "circuitvision.sqlite3"
    monkeypatch.setenv("CV_DATABASE_PATH",str(path))
    component_db.initialize_database()
    return path


def test_database_is_relational_and_has_all_unified_classes(catalog):
    stats=component_db.database_status()
    seed=json.loads(component_db.SEED.read_text())
    assert stats["ready"] is True
    assert stats["categories"] == len(seed["categories"])
    assert stats["components"] == len(seed["components"])
    assert stats["mapped_model_2_classes"] == 87
    assert stats["foreign_keys_valid"] is True
    with component_db.connection() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO model_classes(model_code,class_id,component_id) VALUES ('bogus',999,999)")


def test_component_lookup_reflects_actual_team_guide(catalog):
    row=component_db.get_component("rEsIsToR")
    assert row["reference_source"]=="sqlite"
    assert row["name"]=="Resistor"
    assert "ohms" in row["learn"].lower()
    assert "multimeter" in row["value"].lower()
    assert component_db.get_component("does not exist") is None
    assert component_db.get_component("LCD OLED")["name"]=="Display / indicator"


def test_catalog_search_and_update(catalog):
    before=component_db.list_components("LED",limit=5)
    assert len(before)>0
    assert len(before)<=5
    value=component_db.upsert_component("Classroom Demo Component","general")
    assert value["label"]=="Classroom Demo Component"
    assert value["information_level"]=="general"
    updated=component_db.upsert_component("Classroom Demo Component","resistor")
    assert updated["name"]=="Resistor"
    assert component_db.get_component("Classroom Demo Component")["category"]=="resistor"


def test_real_api_exposes_read_only_database_catalog(catalog):
    client=TestClient(main.app)
    health=client.get("/api/health")
    assert health.status_code==200
    data=health.json()
    assert data["database_ready"] is True
    assert data["database_model_2_class_links"]==87
    assert data["database_component_count"]>=87
    result=client.get("/api/components/LED%20Green")
    assert result.status_code==200
    assert result.json()["reference_source"]=="sqlite"
    assert result.json()["name"]=="LED"
    search=client.get("/api/components",params={"query":"Resistor","limit":3})
    assert search.status_code==200
    assert search.json()["count"]<=3
    assert all("Resistor" in x["label"] for x in search.json()["items"])
    assert client.get("/api/components/nonexistent_zzzz").status_code==404
    assert client.post("/api/components/LED",json={}).status_code==405


def test_model_detection_joins_component_data_without_storing_image(catalog,monkeypatch):
    def fake_inference(img,model,conf,live=False):
        return [{"label":"Resistor","class_id":42,"confidence":.81,"bbox":[.1,.1,.6,.7]}]
    monkeypatch.setattr(main,"infer",fake_inference)
    image=io.BytesIO()
    Image.new("RGB",(100,90),"white").save(image,format="PNG")
    client=TestClient(main.app)
    response=client.post("/api/detect",data={"model":"auto","confidence":".35","live":"true"},
        files={"image":("resistor.png",image.getvalue(),"image/png")})
    assert response.status_code==200,response.text
    payload=response.json()
    assert payload["reference_status"]=="available"
    assert len(payload["detections"])==1
    ref=payload["detections"][0]["component_info"]
    assert ref["reference_source"]=="sqlite"
    assert ref["name"]=="Resistor"
    assert "current" in ref["role"].lower()
    assert list(catalog.parent.iterdir())==[catalog], "Detection must not store user photos"
