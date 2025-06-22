# train_pca_svm.py

import os
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from app.utils.augment_images_from_db import generate_augmented_dataset

# ============ CREAR CARPETA MODELOS ============
os.makedirs("models", exist_ok=True)

# ============ CARGAR DATOS AUMENTADOS ============
X_aug, y_aug = generate_augmented_dataset()

print(f"Total de muestras aumentadas: {len(X_aug)}")
print(f"Dimensión original: {X_aug.shape[1]}")
print(f"Clases únicas: {len(set(y_aug))}")

# ============ PCA ============
pca = PCA(n_components=0.95, svd_solver='full')
X_pca = pca.fit_transform(X_aug)
print(f"Dimensión reducida con PCA: {X_pca.shape[1]}")

# ============ SVM ============
svm = SVC(kernel='linear', probability=True)
svm.fit(X_pca, y_aug)

# ============ GUARDADO ============
joblib.dump(pca, "models/pca_model.pkl")
joblib.dump(svm, "models/svm_model.pkl")

print("✅ PCA y SVM entrenados y guardados en carpeta 'models/'")
