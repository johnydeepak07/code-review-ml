# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

# TestClient wraps your FastAPI app so you can send requests in tests
# without starting a real server
client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'


def test_bad_code_returns_low_score():
    bad_code = (
        "def f(x,y):\n"
        "    for i in range(100):\n"
        "        if i>50:\n"
        "            if x>0:\n"
        "                a=i*3.14\n"
        "    return a"
    )
    response = client.post('/review', json={'code': bad_code})
    assert response.status_code == 200
    data = response.json()
    assert 'readability_score' in data
    assert 'suggestions' in data
    assert data['readability_score'] < 0.6


def test_good_code_returns_high_score():
    good_code = (
        'def calculate_area(width: float, height: float) -> float:\n'
        '    """Calculate the area of a rectangle."""\n'
        '    return width * height\n'
    )
    response = client.post('/review', json={'code': good_code})
    assert response.status_code == 200
    data = response.json()
    assert data['readability_score'] > 0.5


def test_empty_code_returns_400():
    response = client.post('/review', json={'code': '   '})
    assert response.status_code == 400


def test_syntax_error_returns_422():
    response = client.post('/review', json={'code': 'def broken(:\n    pass'})
    assert response.status_code == 422


def test_response_has_all_expected_fields():
    code = 'def add(a, b):\n    """Add two numbers."""\n    return a + b\n'
    response = client.post('/review', json={'code': code})
    data = response.json()
    expected_fields = [
        'readability_score', 'grade', 'cyclomatic_complexity',
        'max_nesting_depth', 'naming_entropy', 'suggestions', 'filename'
    ]
    for field in expected_fields:
        assert field in data, f"Missing field: {field}"