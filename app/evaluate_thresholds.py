import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import random

from app.services.db_operations import get_all_users_with_features
from app.services.face_recognition import euclidean_distance

def generate_verification_pairs(users, n_same=40, n_diff=80):
    same_pairs = []
    diff_pairs = []

    for _ in range(n_same):
        user = random.choice(users)
        vec = user[5]
        same_pairs.append((vec, vec, 1))  # Par positivo

    while len(diff_pairs) < n_diff:
        u1, u2 = random.sample(users, 2)
        if u1[0] != u2[0]:
            diff_pairs.append((u1[5], u2[5], 0))  # Par negativo

    return same_pairs + diff_pairs

def get_scores_and_labels(pairs):
    distances, labels = [], []
    for v1, v2, label in pairs:
        dist = euclidean_distance(v1, v2)
        distances.append(dist)
        labels.append(label)
    return np.array(distances), np.array(labels)

if __name__ == "__main__":
    users = get_all_users_with_features()
    pairs = generate_verification_pairs(users)
    distances, labels = get_scores_and_labels(pairs)

    # ROC y Precision-Recall
    fpr, tpr, _ = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)

    prec, rec, _ = precision_recall_curve(labels, -distances)

    # Visualización
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
