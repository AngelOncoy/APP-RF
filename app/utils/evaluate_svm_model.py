# evaluate_svm_model.py

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from app.utils.augment_images_from_db import generate_augmented_dataset

# ============ CARGAR DATOS ============
X_aug, y_aug = generate_augmented_dataset()

# ============ CARGAR MODELOS ============
pca = joblib.load("models/pca_model.pkl")
svm = joblib.load("models/svm_model.pkl")

# ============ TRANSFORMACIÓN ============
X_pca = pca.transform(X_aug)

# ============ VALIDACIÓN CRUZADA ============
kf = StratifiedKFold(n_splits=min(5, len(np.unique(y_aug))), shuffle=True, random_state=42)
y_pred = cross_val_predict(svm, X_pca, y_aug, cv=kf)

# ============ RESULTADOS ============
print("=== MATRIZ DE CONFUSIÓN ===")
print(confusion_matrix(y_aug, y_pred))

print("\n=== REPORTE DE CLASIFICACIÓN ===")
print(classification_report(y_aug, y_pred, zero_division=0))
