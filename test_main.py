from fastapi.testclient import TestClient

from main import app

client = TestClient(app=app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict() :
    r = client.post("/predict" , json={ "text" : "reveil moi à 5h  stp " })
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["intention"], str)
    assert 0 <= data["confiance"] <= 1
    assert isinstance( data["fiable"] , bool )


def test_alarme() :
    r = client.post("/predict" , json={ "text" : "reveil moi à 5h  stp " })
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["intention"], str)
    assert data["intention"] == "calendar_set"