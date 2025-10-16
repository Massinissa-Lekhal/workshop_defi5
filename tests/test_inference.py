from pathlib import Path
import json

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.model_inference import load_model, predict_image


@pytest.fixture(scope="module")
def tmp_export_model(tmp_path_factory):
    """Crée un export_model factice utilisable par la CI."""
    base = tmp_path_factory.mktemp("export_model")
    (base / "export_model").mkdir(exist_ok=True)
    save_dir = base / "export_model"

    # Mini modèle très simple : 3 classes
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )
    # Sauvegarde "full"
    torch.save(model, save_dir / "hp_cnn_full.pt")

    # Metadata
    meta = {
        "classes": ["gryffindor", "slytherin", "ravenclaw"],
        "input_size": [224, 224],
        "normalize_mean": [0.5, 0.5, 0.5],
        "normalize_std": [0.5, 0.5, 0.5],
    }
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Image factice
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img.save(save_dir / "sample.jpg")

    return save_dir


def test_model_loads(tmp_export_model):
    model, classes, input_size, mean, std = load_model(base_dir=tmp_export_model)
    assert model is not None
    assert isinstance(classes, list)
    assert len(input_size) == 2
    assert len(mean) == 3 and len(std) == 3


def test_fake_prediction(tmp_export_model):
    model, classes, input_size, mean, std = load_model(base_dir=tmp_export_model)
    img_path = Path(tmp_export_model) / "sample.jpg"
    idx = predict_image(model, img_path, input_size, mean, std)
    assert isinstance(idx, int)
    assert 0 <= idx < 3
