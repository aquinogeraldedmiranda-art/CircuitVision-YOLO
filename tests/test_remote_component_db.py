"""Checks the *deployed* CircuitVision SQLite reference integration."""
from __future__ import annotations
import json
import os
import urllib.request

BASE=os.environ.get("CIRCUITVISION_BASE","https://circuitvision-yolo-lab.onrender.com")
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=150) as response:
        assert response.status==200
        return json.load(response)

health=get("/api/health")
assert health.get("database_ready") is True,health
assert health.get("database_component_count",0)>=87,health
assert health.get("database_model_2_class_links")==87,health
assert health.get("yolov11_1_available") is True,health
assert health.get("yolov11_2_available") is True,health
print("PRODUCTION HEALTH:",json.dumps({k:health.get(k) for k in [
    "deployment_commit","database_ready","database_component_count",
    "database_model_2_class_links","yolov11_1_available","yolov11_2_available"
]}),flush=True)

res=get("/api/components/Resistor")
assert res.get("reference_source")=="sqlite",res
assert res.get("name")=="Resistor",res
assert "current" in res.get("role","").lower(),res
assert res.get("information_level")=="category",res
oled=get("/api/components/LCD%20OLED")
assert oled.get("category")=="display",oled
listing=get("/api/components?query=Resistor&limit=10")
assert listing.get("source")=="sqlite" and 1<=listing.get("count",0)<=10,listing
print("PRODUCTION DB REFERENCES:",json.dumps({
    "resistor":res["name"],"oled":oled["name"],"resistor_matches":listing["count"]
}),flush=True)
print("Live SQLite database joins and catalogue retrieval: PASS",flush=True)
