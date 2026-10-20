"""CircuitVision serves updated copy to phones after a new Render release.

These HTTP response tests prevent accidental reuse of an outdated frontend.
"""
from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse
from backend import main

def test_root_html_has_release_specific_script_and_no_cache():
    response=main.home()
    assert isinstance(response,HTMLResponse)
    body=response.body.decode()
    revision=main.frontend_revision()
    assert revision and body.count(revision)>=4
    assert '__CV_ASSET_VERSION__' not in body
    assert f'./app.js?v={revision}' in body
    assert f'./styles.css?v={revision}' in body
    assert f'./config.js?v={revision}' in body
    assert response.headers['cache-control'].startswith('no-store')

def test_frontend_files_are_no_store():
    for path in ('app.js','styles.css','config.js','api.js','motion_tracker.mjs','component_details.mjs'):
        response=main.frontend_asset(path)
        assert isinstance(response,FileResponse)
        assert 'no-store' in response.headers['cache-control']
    assert isinstance(main.frontend_asset('index.html'),HTMLResponse)

def test_no_stale_claims_in_website_copy():
    source=(main.ROOT/'app.js').read_text(encoding='utf-8')
    assert 'The initial live detector uses a public EESOB' not in source
    assert 'team\\\'s own trained YOLO11s model after validation' not in source
    assert 'The initial live detector' not in source
    assert 'projectProgressSection()' in source
