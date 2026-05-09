"""End-to-end tests for the web_control.py REST API.

Uses Flask's test_client so we don't need a real Pi or camera. Verifies:
  - Server starts without Pi (--no-pi mode is implicit when pi_client is None)
  - /api/status returns valid JSON shape
  - /api/destination accepts both single-point and waypoint-list bodies
  - /api/go fails with 400 when no destination set, succeeds otherwise
  - /api/stop always works and clears nav state
  - /api/maps_key returns the configured key (or empty string)
  - Multi-waypoint (A → B) flow ends with two waypoints staged
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force "no pi, no camera, no yolo" by disabling the imports' side effects.
# We import web_control after preparing globals so it doesn't try to talk to
# real hardware.
import web_control as wc


@pytest.fixture
def client():
    """Flask test client with all real I/O disabled."""
    wc.pi_client = None
    wc.camera = None
    wc.recorder = None
    wc.yolo = None
    wc.nav_active = False
    with wc.nav_lock:
        wc.nav.stop()
    with wc.app.test_client() as c:
        yield c


# ============== /api/status ================================================
def test_status_returns_disconnected_when_no_pi(client):
    res = client.get('/api/status')
    assert res.status_code == 200
    j = res.get_json()
    assert j['connected'] is False
    assert j['gps']['valid'] is False
    assert 'nav' in j and j['nav']['state'] == 'IDLE'


def test_status_includes_yolo_section(client):
    j = client.get('/api/status').get_json()
    assert 'yolo' in j
    assert 'person_detected' in j['yolo']


def test_status_reports_no_active_nav_initially(client):
    j = client.get('/api/status').get_json()
    assert j['nav']['active'] is False
    assert j['nav']['waypoints'] == []


# ============== /api/destination ===========================================
def test_destination_single_point(client):
    res = client.post('/api/destination',
                      data=json.dumps({'lat': 33.652, 'lon': 73.161}),
                      content_type='application/json')
    assert res.status_code == 200
    assert res.get_json() == {'ok': True, 'waypoints': 1}

    j = client.get('/api/status').get_json()
    assert len(j['nav']['waypoints']) == 1
    assert abs(j['nav']['waypoints'][0]['lat'] - 33.652) < 1e-6
    assert abs(j['nav']['waypoints'][0]['lon'] - 73.161) < 1e-6


def test_destination_waypoint_list_a_then_b(client):
    """User taps A=pickup then B=destination → two-waypoint nav."""
    body = {'waypoints': [
        {'lat': 33.652, 'lon': 73.160},   # A
        {'lat': 33.653, 'lon': 73.162},   # B
    ]}
    res = client.post('/api/destination', json=body)
    assert res.status_code == 200
    assert res.get_json()['waypoints'] == 2
    j = client.get('/api/status').get_json()
    assert len(j['nav']['waypoints']) == 2
    assert j['nav']['state'] == 'CALIBRATING'   # state machine started


def test_destination_replaces_previous(client):
    client.post('/api/destination', json={'lat': 33.6, 'lon': 73.0})
    client.post('/api/destination', json={'lat': 33.7, 'lon': 73.1})
    j = client.get('/api/status').get_json()
    # Most recent overrides — only 1 waypoint
    assert len(j['nav']['waypoints']) == 1
    assert abs(j['nav']['waypoints'][0]['lat'] - 33.7) < 1e-6


def test_destination_rejects_empty_body(client):
    res = client.post('/api/destination', json={})
    assert res.status_code == 400
    assert res.get_json()['ok'] is False


def test_destination_rejects_invalid_waypoints(client):
    res = client.post('/api/destination', json={'waypoints': [{'oops': 1}]})
    assert res.status_code == 400


def test_destination_skips_invalid_in_list(client):
    """Mixed valid + invalid: keeps valid ones, rejects all if none valid."""
    res = client.post('/api/destination', json={'waypoints': [
        {'lat': 33.6, 'lon': 73.0},
        {'oops': 'bad'},
    ]})
    # We accept the one valid waypoint
    assert res.status_code == 200
    assert res.get_json()['waypoints'] == 1


# ============== /api/go ====================================================
def test_go_fails_without_destination(client):
    res = client.post('/api/go')
    assert res.status_code == 400
    j = res.get_json()
    assert j['ok'] is False
    assert 'no destination' in j.get('error', '').lower()


def test_go_succeeds_after_destination(client):
    client.post('/api/destination', json={'lat': 33.65, 'lon': 73.16})
    res = client.post('/api/go')
    assert res.status_code == 200
    assert res.get_json()['ok'] is True
    j = client.get('/api/status').get_json()
    assert j['nav']['active'] is True


# ============== /api/stop ==================================================
def test_stop_works_even_when_idle(client):
    res = client.post('/api/stop')
    assert res.status_code == 200
    assert res.get_json()['ok'] is True


def test_stop_clears_active_nav(client):
    client.post('/api/destination', json={'lat': 33.65, 'lon': 73.16})
    client.post('/api/go')
    client.post('/api/stop')
    j = client.get('/api/status').get_json()
    assert j['nav']['active'] is False
    assert j['nav']['state'] == 'IDLE'


# ============== /api/maps_key ==============================================
def test_maps_key_returns_string(client):
    """secrets.yaml may or may not be present — just check shape."""
    res = client.get('/api/maps_key')
    assert res.status_code == 200
    j = res.get_json()
    assert 'key' in j
    assert isinstance(j['key'], str)


# ============== / (HTML page) ==============================================
def test_root_serves_html_with_map_div(client):
    res = client.get('/')
    assert res.status_code == 200
    body = res.data.decode()
    # Smoke check that the new UI elements are there
    assert 'id="map"' in body
    assert 'btnLocate' in body
    assert 'searchInput' in body
    assert 'btnPickA' in body
    assert 'btnPickB' in body
    assert 'btnUseCar' in body
    assert 'libraries=places' in body


# ============== Round trip — A → B → GO → STOP ============================
def test_full_a_to_b_flow(client):
    # 1. Start: idle
    j = client.get('/api/status').get_json()
    assert j['nav']['state'] == 'IDLE'
    assert j['nav']['active'] is False

    # 2. Stage two waypoints (user tapped A then B)
    client.post('/api/destination', json={'waypoints': [
        {'lat': 33.6520, 'lon': 73.1613},
        {'lat': 33.6530, 'lon': 73.1620},
    ]})
    j = client.get('/api/status').get_json()
    assert len(j['nav']['waypoints']) == 2

    # 3. GO
    assert client.post('/api/go').get_json()['ok'] is True
    j = client.get('/api/status').get_json()
    assert j['nav']['active'] is True

    # 4. STOP
    assert client.post('/api/stop').get_json()['ok'] is True
    j = client.get('/api/status').get_json()
    assert j['nav']['active'] is False
    assert j['nav']['state'] == 'IDLE'
    assert j['nav']['waypoints'] == []


# ============== Obstacle override during navigation =====================
def test_obstacle_override_stops_for_close_person(client):
    from ml.actions import FORWARD, STOP
    sensors = {'FL': 200, 'FR': 200, 'FW': 200,
               'BC': 200, 'LS': 200, 'RS': 200}
    yolo = {'person_detected': 1, 'object_detected': 0,
            'nearest_area_ratio': 0.5, 'nearest_position_x': 0.0,
            'num_objects': 1}
    final, reason = wc._apply_obstacle_override(FORWARD, sensors, yolo)
    assert final == STOP
    assert 'OBST_STOP' in reason


def test_obstacle_override_passes_through_when_clear(client):
    from ml.actions import FORWARD
    sensors = {'FL': 300, 'FR': 300, 'FW': 300,
               'BC': 300, 'LS': 300, 'RS': 300}
    yolo = {'person_detected': 0, 'object_detected': 0,
            'nearest_area_ratio': 0.0, 'nearest_position_x': 0.0,
            'num_objects': 0}
    final, reason = wc._apply_obstacle_override(FORWARD, sensors, yolo)
    assert final == FORWARD
    assert reason == ''


def test_obstacle_override_reverses_when_front_blocked(client):
    """Front emergency triggers hybrid REVERSE → override should respect it."""
    from ml.actions import FORWARD, REVERSE, REVERSE_LEFT, REVERSE_RIGHT
    sensors = {'FL': 20, 'FR': 20, 'FW': 20,
               'BC': 200, 'LS': 100, 'RS': 100}
    yolo = {'person_detected': 0, 'object_detected': 0,
            'nearest_area_ratio': 0.0, 'nearest_position_x': 0.0,
            'num_objects': 0}
    final, reason = wc._apply_obstacle_override(FORWARD, sensors, yolo)
    assert final in (REVERSE, REVERSE_LEFT, REVERSE_RIGHT)
    assert 'OBST_REVERSE' in reason


def test_obstacle_override_handles_empty_sensors(client):
    """No sensor data yet → don't override, let nav action through."""
    from ml.actions import FORWARD
    yolo = {'person_detected': 0, 'object_detected': 0,
            'nearest_area_ratio': 0.0, 'nearest_position_x': 0.0,
            'num_objects': 0}
    final, reason = wc._apply_obstacle_override(FORWARD, {}, yolo)
    assert final == FORWARD
    assert reason == ''
