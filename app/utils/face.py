# utils/face.py
import torch, yaml, dlib, cv2, numpy as np
from pathlib import Path
from torchvision.transforms import functional as TF

CFG = yaml.safe_load(open(Path(__file__).parent.parent /
                          "models/train_config.yaml"))
IMG_SIZE = CFG["face_size"]
THRESH   = CFG["threshold"]

# 1. detector y alineador (dlib)
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_5pt.dat")

# 2. embedder (TorchScript, CPU)
embedder  = torch.jit.load("models/face_embedder.pt", map_location="cpu").eval()

def align_face(bgr: np.ndarray):
    dets = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1)
    if not dets:  return None
    shape = predictor(bgr, dets[0])
    chip  = dlib.get_face_chip(bgr, shape, IMG_SIZE)
    return cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)

@torch.inference_mode()
def get_embedding(gray: np.ndarray):
    x = TF.to_tensor(gray).unsqueeze(0)      # (1,1,128,128)
    x = TF.normalize(x, [0.5], [0.5])
    return embedder(x)[0].numpy()            # 128-D L2-normalizado
