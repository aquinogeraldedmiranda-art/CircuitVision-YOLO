"""Project progress is committed alongside deployed website code.

These tests check provenance and API wiring. They do not establish model
recognition accuracy or imply that an unfinished Colab run is complete.
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app

ROOT=Path(__file__).resolve().parents[1]


def test_reviewed_project_milestones_are_truthful():
    progress=json.loads((ROOT/'project_status.json').read_text('utf-8'))
    assert progress['schema_version']==1
    assert progress['project']=='CircuitVision'
    assert progress['reviewed_on']
    ids={item['id']:item for item in progress['milestones']}
    assert len(ids)==len(progress['milestones'])
    assert ids['website']['status']=='deployed'
    assert ids['sam']['status']=='integration_prepared'
    assert ids['unified']['status']=='awaiting_results'
    assert 'not' in progress['limitations'].lower()
    assert len(progress['recent_updates'])>0


def test_project_status_endpoint_is_fresh_and_reports_actual_runtime():
    response=TestClient(app).get('/api/project-status')
    assert response.status_code==200
    assert response.headers['cache-control']=='no-store'
    data=response.json()
    assert data['project']['project']=='CircuitVision'
    runtime=data['runtime']
    assert type(runtime['model_ready']) is bool
    assert type(runtime['sam_ready']) is bool
    assert type(runtime['round3_ready']) is bool
    assert type(runtime['unified_model_active']) is bool
    assert isinstance(runtime['class_count'],int)
    # Do not advertise a jointly trained checkpoint while a 57-class
    # student detector remains the production model.
    if runtime['model_ready'] and runtime['class_count']==57:
        assert not runtime['unified_model_active']
