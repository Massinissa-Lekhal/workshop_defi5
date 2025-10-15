import torch, json
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

def load_model():
    save_dir = Path("export_model")
    full_path = save_dir / "hp_cnn_full.pt"
    state_path = save_dir / "hp_cnn_state.pth"
    meta_path  = save_dir / "metadata.json"

    assert full_path.exists() or state_path.exists(), "Aucun modèle sauvegardé trouvé."
    assert meta_path.exists(), "metadata.json introuvable."

    with open(meta_path, "r") as f:
        meta = json.load(f)

    classes = meta.get("classes", [])
    input_size = tuple(meta.get("input_size", [224,224]))
    mean = meta.get("normalize_mean", [0.5,0.5,0.5])
    std  = meta.get("normalize_std",  [0.5,0.5,0.5])

    try:
        model = torch.load(full_path, map_location="cpu", weights_only=False)
    except Exception:
        from torch.serialization import add_safe_globals
        import torch.nn as nn
        add_safe_globals([nn.Sequential, nn.modules.container.Sequential])
        model = torch.load(full_path, map_location="cpu", weights_only=False)

    model.eval()
    return model, classes, input_size, mean, std


def predict_image(model, image_path, input_size, mean, std):
    transform = T.Compose([
        T.Resize(input_size),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        pred_idx = int(probs.argmax().item())
        return pred_idx
