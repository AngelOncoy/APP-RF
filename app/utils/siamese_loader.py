import json
import torch
import os
import torch.nn as nn
import torch.nn.functional as F


# ------------------------ EmbeddingNet --------------------------
class EmbeddingNetV2(nn.Module):
    def __init__(self):
        super(EmbeddingNetV2, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


# ------------------------ SiameseNetwork --------------------------
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2

    def get_embedding(self, x):
        return self.embedding_net(x)


# ------------------------ Singleton Loader --------------------------
_model = None
_threshold = None

def get_siamese_model(device: str = "cpu"):
    global _model, _threshold

    if _model is None or _threshold is None:
        # Ruta absoluta segura al archivo actual
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "siamese_face_modelV2.pth")
        threshold_path = os.path.join(current_dir, "optimal_thresholdV2.json")

        embedding_net = EmbeddingNetV2()
        model = SiameseNetwork(embedding_net)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with open(threshold_path, "r") as f:
            threshold = json.load(f)["threshold"]

        _model = model
        _threshold = threshold
        print("✅ Modelo Siamese cargado correctamente.")

    return _model, _threshold
