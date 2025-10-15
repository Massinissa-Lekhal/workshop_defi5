from src.model_inference import load_model, predict_image
from pathlib import Path

def test_model_loads():
    # doit au moins charger les fichiers existants
    model, classes, input_size, mean, std = load_model()
    assert model is not None
    assert isinstance(classes, list)
    assert len(input_size) == 2

def test_fake_prediction():
    model, classes, input_size, mean, std = load_model()
    img_path = Path("export_model/sample.jpg")
    if not img_path.exists():
        return  # ignorer si pas d'image
    idx = predict_image(model, img_path, input_size, mean, std)
    assert isinstance(idx, int)
