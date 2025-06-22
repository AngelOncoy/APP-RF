import json
import tkinter as tk
from tkinter import filedialog, messagebox

from app.services.face_recognition import extract_face_features, euclidean_distance
from app.services.db_operations    import get_all_users_with_features


# ---------- CORE -------------------------------------------------
TOP_K = 5   # cuántos mostrar en modo verbose


def get_vectors_from_db():
    """Regresa [(label_str, vec_list[float]), …]"""
    data = []
    for uid, name, last, email, req, vec in get_all_users_with_features():
        label = f"{uid}  |  {name} {last}"
        data.append((label, vec))
    return data


def run_best_match(image_path: str):
    vec_ref = json.loads(extract_face_features(image_path))
    best_label, best_sim = "—", -1.0
    for label, vec_db in get_vectors_from_db():
        sim = euclidean_distance(vec_ref, vec_db)
        if sim > best_sim:
            best_sim, best_label = sim, label
    return best_label, best_sim


def run_verbose(image_path: str, k: int = TOP_K):
    vec_ref = json.loads(extract_face_features(image_path))
    sims = []
    for label, vec_db in get_vectors_from_db():
        sim = euclidean_distance(vec_ref, vec_db)
        sims.append((sim, label))
    sims.sort(reverse=True)
    return sims[:k]


# ---------- GUI --------------------------------------------------
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
    label, sim = run_best_match(path)
    msg = f"Mejor coincidencia:\n\n{label}\n\nSimilitud: {sim:.3f}"
    messagebox.showinfo("Resultado", msg)


def gui_verbose(path):
    top = run_verbose(path)
    msg_lines = [f"{i+1}.  {lab}\n    sim = {s:.4f}"
                 for i, (s, lab) in enumerate(top)]
    messagebox.showinfo("Verbose Top-k", "\n\n".join(msg_lines))


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
