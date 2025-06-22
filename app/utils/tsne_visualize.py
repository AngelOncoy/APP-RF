# tsne_visualize.py

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from app.services.db_operations import get_all_users_with_features

# ========= CARGAR VECTORES =========
users = get_all_users_with_features()

X = []
y = []

for user_id, _, _, _, _, vec in users:
    X.append(vec)
    y.append(str(user_id))  # Aseguramos que user_id sea string

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"Total muestras: {len(X)} — Dimensión: {X.shape[1]} — Clases únicas: {len(set(y))}")

# ========= ENCODING DE CLASES =========
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========= APLICAR t-SNE =========
print("Aplicando t-SNE...")
tsne = TSNE(n_components=2, perplexity=10, learning_rate=100, max_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X)

# ========= VISUALIZACIÓN =========
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, cmap="tab20", alpha=0.8, s=25)
plt.colorbar(scatter, label="Clase (user_id codificado)")
plt.title("t-SNE: Visualización 2D de vectores faciales (sin PCA)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.grid(True)
plt.tight_layout()
plt.show()
