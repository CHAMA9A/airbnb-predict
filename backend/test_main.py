from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

# Test de l'endpoint racine
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "API OK"}

# Test de l'endpoint /predict avec des données valides
def test_predict_valid():
    data = {
        "neighbourhood_group": "Manhattan",
        "room_type": "Entire home/apt",
        "minimum_nights": 3,
        "number_of_reviews": 12,
        "availability_365": 200,
        "latitude": 40.735,
        "longitude": -73.99,
        "reviews_per_month": 0.8,
        "last_review_year": 2023,
        "last_review_month": 5,
        "last_review_dayofweek": 4
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "high_demand" in response.json()
    assert "probability" in response.json()

# Test avec données manquantes (FastAPI doit retourner une erreur 422)
def test_predict_missing_field():
    data = {
        # Il manque "neighbourhood_group"
        "room_type": "Entire home/apt",
        "minimum_nights": 3,
        "number_of_reviews": 12,
        "availability_365": 200,
        "latitude": 40.735,
        "longitude": -73.99,
        "reviews_per_month": 0.8,
        "last_review_year": 2023,
        "last_review_month": 5,
        "last_review_dayofweek": 4
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 422
