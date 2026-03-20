from ultralytics import YOLO
from pathlib import Path
import urllib.request


MODEL_URL = "https://drive.usercontent.google.com/download?id=13Kylz4k_Rs3KUUF3KU_QTv2v-pJ-jKQw&export=download&authuser=0&confirm=t&uuid=344b8400-0108-4a4a-a0ab-3ad6ab532041&at=AGN2oQ0vIEEJFtRtbdRmXOfW-RTD%3A1773969961842"



def load_model(model_path: str | None = None):
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "weights" / "model_hifi_5k.pt"
    model_path = Path(model_path)

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[trueSV] Downloading the model ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    model = YOLO(str(model_path))


    return model