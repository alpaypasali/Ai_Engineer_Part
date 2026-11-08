import pytest
import subprocess
import sys
import shutil

from pathlib import Path
from predict import predict_news


def test_empty_input():

    with pytest.raises(ValueError, match="Haber metni boş olamaz"):
        predict_news("")

    print("Test 1 — Empty Input: PASSED")


def test_whitespace_input():

    with pytest.raises(ValueError, match="Haber metni boş olamaz"):
        predict_news("     ")

    print("Test 2 — Whitespace Input: PASSED")


def test_missing_model(tmp_path):

    project_dir = Path(__file__).resolve().parents[1]

    original_script = project_dir / "predict.py"
    temporary_script = tmp_path / "predict.py"

    shutil.copyfile(original_script, temporary_script)

    result = subprocess.run(
        [
            sys.executable,
            str(temporary_script),
            "NASA announces new satellite"
        ],
        capture_output=True,
        text=True
    )

    assert result.returncode != 0
    assert "FileNotFoundError" in result.stderr
    assert "tfidf_pipeline.joblib" in result.stderr
    assert "Predicted Class:" not in result.stdout

    print("Test 3 — Missing Model: PASSED")


def test_none_input():

    with pytest.raises(TypeError, match="Haber metni string olmalıdır"):
        predict_news(None)

    print("Test 4 — None Input: PASSED")


def test_integer_input():

    with pytest.raises(TypeError, match="Haber metni string olmalıdır"):
        predict_news(123)

    print("Test 5 — Integer Input: PASSED")


def test_valid_input():

    prediction = predict_news("NASA launches satellite")

    assert prediction in [
        "World",
        "Sports",
        "Business",
        "Sci/Tech"
    ]

    print("Test 6 — Valid Input: PASSED")