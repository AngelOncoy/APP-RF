import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import random

from app.services.db_operations import get_all_users_with_features
from app.services.face_recognition import euclidean_distance

def generate_verification_pairs(users, n_same=40, n_diff=80):
    same_pairs = []
    diff_pairs = []

    for _ in range(n_same):
        user = random.choice(users)
        vec = user[5]
        same_pairs.append((vec, vec, 1))

    while len(diff_pairs) < n_diff:
        u1, u2 = random.sample(users, 2)
        if u1[0] != u2[0]:
            diff_pairs.append((u1[5], u2[5], 0))

    return same_pairs + diff_pairs

def get_scores_and_labels(pairs):
    distances, labels = [], []
    for v1, v2, label in pairs:
        dist = euclidean_distance(v1, v2)
        distances.append(dist)
        labels.append(label)
    return np.array(distances), np.array(labels)

def find_best_threshold(distances, labels):
    thresholds = np.linspace(0.1, 1.0, 100)
    f1_scores = []

    for t in thresholds:
        preds = distances <= t
        score = f1_score(labels, preds)
        f1_scores.append(score)

    best_idx = int(np.argmax(f1_scores))
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, thresholds, f1_scores

if __name__ == "__main__":
    users = get_all_users_with_features()
    pairs = generate_verification_pairs(users)
    distances, labels = get_scores_and_labels(pairs)

    best_threshold, best_f1, thresholds, f1_scores = find_best_threshold(distances, labels)

    print(f"✅ Mejor umbral encontrado: {best_threshold:.4f}")
    print(f"🎯 F1-score en ese umbral: {best_f1:.4f}")

    # Gráfica F1-score vs Umbral
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, label="F1-score")
    plt.axvline(best_threshold, color='r', linestyle='--', label=f"Mejor umbral = {best_threshold:.3f}")
    plt.xlabel("Umbral de Distancia Euclidiana")
    plt.ylabel("F1-score")
    plt.title("Selección del Mejor Umbral")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
