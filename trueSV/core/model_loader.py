from ultralytics import YOLO
from pathlib import Path
import urllib.request


MODEL_URL = "https://drive.usercontent.google.com/download?id=17jDHlwMyQHtjE269QVe5yaGd9tkY58cX&export=download&authuser=0&confirm=t&uuid=7733230b-b097-40ba-93cf-3cdd54dee221&at=AGN2oQ2mPF99EDpCtjV3YHwl4UDR%3A1773260064723"



def load_model(model_path: str | None = None):
    if model_path is None:
        model_path = Path(__file__).resolve().parent.parent / "weights" / "model_hifi_5k.pt"
    model_path = Path(model_path)

    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[trueSV] Downloading hifi model 1 ...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    model = YOLO(str(model_path))


    return model