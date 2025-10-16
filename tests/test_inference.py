from pathlib import Path

from src.model_inference import load_model, predict_image


def test_model_loads():
    model, classes, input_size, mean, std = load_model()
    assert model is not None
    assert isinstance(classes, list)
    assert len(input_size) == 2


def test_fake_prediction():
    model, classes, input_size, mean, std = load_model()
    img_path = Path("export_model/sample.jpg")
    if not img_path.exists():
        # pas d'image de test => on n'échoue pas le build
        return
    idx = predict_image(model, img_path, input_size, mean, std)
    assert isinstance(idx, int)
