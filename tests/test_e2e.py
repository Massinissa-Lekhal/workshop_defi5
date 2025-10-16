#import pytest
#from pathlib import Path
#from src.model_inference import load_model, predict_image


#@pytest.mark.e2e
#def test_e2e_prediction(tmp_export_model):
#    model, classes, input_size, mean, std = load_model(base_dir=tmp_export_model)
#    img_path = Path(tmp_export_model) / "sample.jpg"
#    idx = predict_image(model, img_path, input_size, mean, std)
#    assert isinstance(idx, int)
