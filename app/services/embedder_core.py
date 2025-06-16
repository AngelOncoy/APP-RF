"""
Carga única: detector dlib + alineador + Tiny-FaceNet (TorchScript)
y expone   align_face(bgr)  y  embed_face(gray).
"""
import dlib, cv2, torch, yaml, numpy as np
from pathlib import Path
from torchvision.transforms import functional as TF

MODELS = Path(__file__).parent.parent / "models"
CFG    = yaml.safe_load(open(MODELS / "train_config.yaml"))

# ─ cargar artefactos ─────────────────────────────────────────────
_detector  = dlib.get_frontal_face_detector()
# embedder_core.py  (solo cambia estas dos líneas)

_predictor = dlib.shape_predictor(str(MODELS / "shape_5pt.dat"))
#                                           ^^^ convierte a str

_embedder  = torch.jit.load(str(MODELS / "face_embedder.pt"),
                            map_location="cpu").eval()
#                      ^^^ idem

IMG_SIZE   = CFG["face_size"]

def align_face(bgr: np.ndarray) -> np.ndarray | None:
    dets = _detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1)
    if not dets:
        return None
    shp  = _predictor(bgr, dets[0])
    chip = dlib.get_face_chip(bgr, shp, IMG_SIZE)
    return cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)

@torch.inference_mode()
def embed_face(gray: np.ndarray) -> np.ndarray:
    x = TF.to_tensor(gray).unsqueeze(0)      # (1,1,128,128)
    x = TF.normalize(x, [0.5], [0.5])
    return _embedder(x)[0].numpy()           # 128-D  L2-norm
