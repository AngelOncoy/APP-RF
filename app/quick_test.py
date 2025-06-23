import json
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

from app.services.face_recognition import extract_face_features, euclidean_distance, FACE_CASCADE
from app.services.db_operations import get_all_users_with_features, get_user_image

DISTANCE_THRESHOLD = 0.45
TOP_K = 5

# -------------- Obtener vectores -------------------
def get_vectors_from_db():
    data = []
    for uid, name, last, email, req, vec in get_all_users_with_features():
        label = f"{uid}  |  {name} {last}"
        data.append((label, uid, vec))
    return data

# -------------- Extraer rostro visible -------------------
def extract_face_from_image_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    return img  # fallback: imagen entera

def extract_face_from_path(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y+h, x:x+w]
    return img

# -------------- Mostrar rostro en ventana emergente -------------------
def show_faces(query_img, match_img, label_text):
    win = tk.Toplevel()
    win.title("Visualización Comparación")
    win.geometry("600x300")

    tk.Label(win, text=label_text, font=("Helvetica", 12)).pack()

    query = cv2.resize(query_img, (200, 200))
    match = cv2.resize(match_img, (200, 200))

    qtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(query, cv2.COLOR_BGR2RGB)))
    mtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(match, cv2.COLOR_BGR2RGB)))

    tk.Label(win, text="Imagen ingresada").pack(side="left", padx=10)
    tk.Label(win, image=qtk).pack(side="left", padx=10)
    tk.Label(win, text="Mejor coincidencia").pack(side="left", padx=10)
    tk.Label(win, image=mtk).pack(side="left", padx=10)

    win.mainloop()

# -------------- Comparaciones -------------------
def run_best_match(image_path: str):
    vec_ref = json.loads(extract_face_features(image_path))
    best_label, best_dist, best_uid = "—", float("inf"), None

    for label, uid, vec_db in get_vectors_from_db():
        dist = euclidean_distance(vec_ref, vec_db)
        if dist < best_dist:
            best_dist, best_label, best_uid = dist, label, uid

    match = best_dist <= DISTANCE_THRESHOLD
    return best_label, best_dist, match, best_uid

def run_verbose(image_path: str, k: int = TOP_K):
    vec_ref = json.loads(extract_face_features(image_path))
    distances = []
    for label, uid, vec_db in get_vectors_from_db():
        dist = euclidean_distance(vec_ref, vec_db)
        distances.append((dist, label, uid))
    distances.sort()
    return distances[:k]

# -------------- GUI helpers -------------------
def pick_image_and(func):
    path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not path:
        return
    try:
        func(path)
    except Exception as err:
        messagebox.showerror("Error", str(err))

def gui_best_match(path):
    label, dist, match, uid = run_best_match(path)
    msg = f"Mejor coincidencia:\n\n{label}\n\nDistancia: {dist:.4f}\n\n"
    msg += "✅ Coincidencia dentro del umbral" if match else "❌ No coincide (fuera del umbral)"

    img_q = extract_face_from_path(path)
    img_m_bytes = get_user_image(uid)
    img_m = extract_face_from_image_bytes(img_m_bytes)

    show_faces(img_q, img_m, msg)

def gui_verbose(path):
    top = run_verbose(path)
    img_q = extract_face_from_path(path)

    for rank, (dist, label, uid) in enumerate(top, 1):
        img_m_bytes = get_user_image(uid)
        img_m = extract_face_from_image_bytes(img_m_bytes)
        label_text = f"{rank}. {label}\nDistancia = {dist:.4f}"
        show_faces(img_q, img_m, label_text)

# -------------- INTERFAZ -------------------
root = tk.Tk()
root.title("Quick Test – Comparador Facial")
root.geometry("460x230")

tk.Label(root, text="Quick Test de Comparación", font=("Helvetica", 16)).pack(pady=20)

tk.Button(root,
          text="Comparar (mejor match)",
          font=("Helvetica", 12),
          command=lambda: pick_image_and(gui_best_match),
          width=32).pack(pady=5)

tk.Button(root,
          text="Verbose Top-k",
          font=("Helvetica", 12),
          command=lambda: pick_image_and(gui_verbose),
          width=32).pack(pady=5)

root.mainloop()
