import Reconocimiento_Facial
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección Facial con Face Recognition")

        # Variable para umbral (default 0.6)
        self.umbral_distancia = tk.DoubleVar(value=0.6)

        self.frame_umbral = tk.Frame(root)
        self.frame_umbral.pack(pady=5)

        tk.Label(self.frame_umbral, text="Umbral distancia (0.3 - 0.8): ").pack(side=tk.LEFT)
        self.entry_umbral = tk.Entry(self.frame_umbral, textvariable=self.umbral_distancia, width=5)
        self.entry_umbral.pack(side=tk.LEFT)

        self.btn_cargar = tk.Button(root, text="Seleccionar Imagen para Buscar", command=self.cargar_imagen)
        self.btn_cargar.pack(pady=10)

        self.btn_calibrar = tk.Button(root, text="Seleccionar Imagen para Calibrar Umbral",
                                      command=self.calibrar_umbral)
        self.btn_calibrar.pack(pady=10)

        self.lbl_nombre = tk.Label(root, text="Nombre: ")
        self.lbl_nombre.pack()

        self.lbl_correo = tk.Label(root, text="Correo: ")
        self.lbl_correo.pack()

        self.lbl_similitud = tk.Label(root, text="Similitud: ")
        self.lbl_similitud.pack()

        self.text_calibracion = tk.Text(root, height=10, width=50)
        self.text_calibracion.pack(pady=10)

        self.canvas = tk.Canvas(root, width=300, height=300)
        self.canvas.pack(pady=10)

        self.imagen_tk = None

    def cargar_imagen(self):
        try:
            umbral = float(self.entry_umbral.get())
            if not (0.3 <= umbral <= 0.8):
                raise ValueError("Umbral fuera de rango")
        except Exception:
            messagebox.showerror("Error", "Ingresa un umbral válido entre 0.3 y 0.8")
            return

        ruta = filedialog.askopenfilename(title="Seleccionar imagen",
                                          filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")])
        if not ruta:
            return

        resultado, mensaje = Reconocimiento_Facial.buscar_persona_por_imagen(ruta, umbral_distancia=umbral)

        if resultado is None:
            messagebox.showerror("Error", mensaje)
            self.limpiar_resultados()
        else:
            persona, similitud = resultado, mensaje
            self.mostrar_resultados(persona, similitud)

    def calibrar_umbral(self):
        ruta = filedialog.askopenfilename(title="Seleccionar imagen para calibrar",
                                          filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png")])
        if not ruta:
            return

        umbrales = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
        self.text_calibracion.delete("1.0", tk.END)
        self.text_calibracion.insert(tk.END, f"Pruebas de calibración para: {ruta}\n\n")

        for umbral in umbrales:
            resultado, mensaje = Reconocimiento_Facial.buscar_persona_por_imagen(ruta, umbral_distancia=umbral)
            if resultado is None:
                self.text_calibracion.insert(tk.END, f"Umbral {umbral:.2f}: No encontrado. Mensaje: {mensaje}\n")
            else:
                persona, similitud = resultado, mensaje
                self.text_calibracion.insert(tk.END,
                                             f"Umbral {umbral:.2f}: Encontrado {persona['nombre']} {persona['apellido']} con similitud {similitud * 100:.2f}%\n")

    def mostrar_resultados(self, persona, similitud):
        self.lbl_nombre.config(text=f"Nombre: {persona['nombre']} {persona['apellido']}")
        self.lbl_correo.config(text=f"Correo: {persona['correo']}")
        self.lbl_similitud.config(text=f"Similitud: {similitud * 100:.2f}%")

        try:
            imagen = Image.open(persona['foto'])
            imagen = imagen.resize((300, 300), Image.LANCZOS)
            self.imagen_tk = ImageTk.PhotoImage(imagen)
            self.canvas.create_image(150, 150, image=self.imagen_tk)
        except Exception as e:
            messagebox.showwarning("Advertencia", f"No se pudo cargar la imagen de la persona: {e}")
            self.canvas.delete("all")

    def limpiar_resultados(self):
        self.lbl_nombre.config(text="Nombre: ")
        self.lbl_correo.config(text="Correo: ")
        self.lbl_similitud.config(text="Similitud: ")
        self.canvas.delete("all")
        self.text_calibracion.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()