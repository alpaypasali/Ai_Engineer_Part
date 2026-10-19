import subprocess
import sys
from pathlib import Path

import pytest
import predict

from fastapi.testclient import TestClient
from api import app


# ==========================================
# TEST CLIENT
# ==========================================

client = TestClient(app)


# ==========================================
# TEST 1 — EMPTY TEXT
# ==========================================

def test_empty_text():

    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 400


# ==========================================
# TEST 2 — MISSING TEXT FIELD
# ==========================================

def test_missing_text_field():

    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


# ==========================================
# TEST 3 — CLI AND API SAME PREDICTION
# ==========================================

def test_cli_api_same_prediction():

    text = "My mortgage payment is incorrect"

    # Gerçek model dosyasını kontrol et
    assert predict.MODEL_PATH.is_file(), (
        f"Model file not found: {predict.MODEL_PATH}"
    )

    # API üzerinden tahmin
    response = client.post(
        "/predict",
        json={"text": text}
    )

    assert response.status_code == 200

    api_result = response.json()

    # CLI üzerinden tahmin
    project_root = Path(__file__).resolve().parent.parent

    cli_result = subprocess.run(
        [
            sys.executable,
            str(project_root / "predict.py"),
            "--text",
            text
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True
    )

    # CLI çıktısını sözlüğe dönüştür
    cli_output = {}

    for line in cli_result.stdout.splitlines():

        if ": " in line:

            key, value = line.split(": ", 1)

            cli_output[key] = value

    # CLI çıktısının geçerli olduğunu kontrol et
    assert "Category" in cli_output, cli_result.stdout
    assert "Probability" in cli_output, cli_result.stdout
    assert "Status" in cli_output, cli_result.stdout

    # API ve CLI sonuçlarını karşılaştır
    assert api_result["category"] == cli_output["Category"]

    assert api_result["probability"] == float(
        cli_output["Probability"]
    )

    assert api_result["status"] == cli_output["Status"].lower()


# ==========================================
# TEST 4 — HUMAN REVIEW
# ==========================================

def test_human_review(monkeypatch):

    # Sahte model oluştur
    class FakeModel:

        classes_ = [0, 1, 2, 3]

        def predict(self, texts):
            return [3]

        def predict_proba(self, texts):
            return [[0.20, 0.20, 0.20, 0.40]]

    # Gerçek model yerine sahte model kullan
    monkeypatch.setattr(
        predict,
        "model",
        FakeModel()
    )

    # API üzerinden tahmin
    response = client.post(
        "/predict",
        json={"text": "My payment is incorrect"}
    )

    assert response.status_code == 200

    result = response.json()

    # Human Review kontrolü
    assert result["category"] == "Mortgage"

    assert result["probability"] == 0.40

    assert result["status"] == "human review required"


# ==========================================
# TEST 5 — MODEL LOADED ONCE
# ==========================================

def test_model_loaded_once(monkeypatch, tmp_path):

    # Geçici model dosyası oluştur
    fake_path = tmp_path / "fake_model.joblib"

    fake_path.touch()

    # Model yükleme sayacı
    load_count = {"count": 0}

    # Sahte model
    class FakeModel:

        classes_ = [0, 1, 2, 3]

        def predict(self, texts):
            return [3]

        def predict_proba(self, texts):
            return [[0.05, 0.05, 0.10, 0.80]]

    # Sahte model yükleme fonksiyonu
    def fake_load(path):

        load_count["count"] += 1

        return FakeModel()

    # Başlangıçta model yüklü değil
    monkeypatch.setattr(
        predict,
        "model",
        None
    )

    # Geçici model yolunu ayarla
    monkeypatch.setattr(
        predict,
        "MODEL_PATH",
        fake_path
    )

    # joblib.load fonksiyonunu değiştir
    monkeypatch.setattr(
        predict.joblib,
        "load",
        fake_load
    )

    # Birinci API isteği
    response1 = client.post(
        "/predict",
        json={"text": "My mortgage payment is incorrect"}
    )

    # İkinci API isteği
    response2 = client.post(
        "/predict",
        json={"text": "My mortgage payment is incorrect"}
    )

    # İki istek de başarılı olmalı
    assert response1.status_code == 200

    assert response2.status_code == 200

    # Model yalnızca bir kez yüklenmeli
    assert load_count["count"] == 1