import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import json
import cv2
import numpy as np
from app.utils.siamese_loader import load_siamese_model
from app.utils.face_utils import preprocess_image, euclidean_distance

# --- Cargar modelo entrenado ---
MODEL_PATH = "siamese_face_modelV2.pth"
THRESH_PATH = "optimal_thresholdV2.json"
DEVICE = "cpu"
model, THRESHOLD = load_siamese_model(MODEL_PATH, THRESH_PATH, DEVICE)

# --- Base de usuarios cargados manualmente ---
USER_DATABASE = []  # lista de (label, vector, imagen)

# --- GUI principal ---
root = tk.Tk()
root.title("Test Comparación Facial con Modelo Siamese")
root.geometry("500x500")

# Mostrar imagen
img_label = tk.Label(root)
img_label.pack(pady=10)

def show_image(img_path):
    img = Image.open(img_path).convert("L").resize((100, 100))
    img_tk = ImageTk.PhotoImage(img)
    img_label.configure(image=img_tk)
    img_label.image = img_tk

# Seleccionar y agregar imagen base
def cargar_usuario():
    path = filedialog.askopenfilename(title="Agregar usuario", filetypes=[("Imagen", "*.jpg *.png")])
    if not path:
        return
    show_image(path)
    vec = preprocess_image(path, model.embedding_net, DEVICE)
    USER_DATABASE.append((path, vec))
    messagebox.showinfo("Listo", "Usuario agregado a base temporal")

# Comparar contra base
def comparar():
    path = filedialog.askopenfilename(title="Comparar imagen", filetypes=[("Imagen", "*.jpg *.png")])
    if not path:
        return
    show_image(path)
    vec_query = preprocess_image(path, model.embedding_net, DEVICE)

    best_label, best_dist = "", float("inf")
    for label, vec in USER_DATABASE:
        dist = euclidean_distance(vec_query, vec)
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_dist < THRESHOLD:
        msg = f"✅ MATCH con {best_label}\nDistancia: {best_dist:.3f}"
    else:
        msg = f"❌ NO MATCH\nDistancia: {best_dist:.3f}"

    messagebox.showinfo("Resultado", msg)

# Widgets
btn1 = tk.Button(root, text="Agregar imagen base", font=("Helvetica", 12), command=cargar_usuario)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Comparar imagen", font=("Helvetica", 12), command=comparar)
btn2.pack(pady=10)

root.mainloop()
