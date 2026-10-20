"""Internal development ledger remains versioned in GitHub but is not a public web page."""
import json
from pathlib import Path
from backend.main import app

ROOT=Path(__file__).resolve().parents[1]

def test_internal_status_record_remains_reviewed():
    p=json.loads((ROOT/'project_status.json').read_text('utf-8'))
    assert p['schema_version']==1 and p['project']=='CircuitVision'
    assert p['reviewed_on'] and p['milestones']
    assert len({m['id'] for m in p['milestones']})==len(p['milestones'])
    assert p['recent_updates']
    assert all(m['status'] in {'deployed','integration_prepared','awaiting_results',
               'in_progress','locally_tested','validated','blocked'} for m in p['milestones'])

def test_internal_project_history_not_exposed_as_public_api():
    assert '/api/project-status' not in {r.path for r in app.routes}
    js=(ROOT/'app.js').read_text('utf-8')
    assert 'projectProgressSection()' not in js
    assert 'refreshProjectStatus()' not in js
