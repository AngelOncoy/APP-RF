import tkinter as tk
from tkinter import filedialog, messagebox
from app.controllers.face_controller import compare_external_image


# Función para seleccionar imagen y comparar
def select_and_compare():
    image_path = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imagenes", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        return

    # Llamamos a la función de comparación
    try:
        result = compare_external_image(image_path)

        if result['match']:
            user = result['user_data']
            msg = f"✅ MATCH: {user['name']} {user['last_name']} (Similarity: {result['similarity']:.3f})\n"

            if user['requisitioned']:
                msg += "\n🚨 ALERTA DE SEGURIDAD: Usuario requisitoriado 🚨"
            else:
                msg += "\nUsuario permitido."

        else:
            msg = f"❌ NO MATCH (Similarity: {result['similarity']:.3f})\nUsuario desconocido."

        messagebox.showinfo("Resultado de Comparación", msg)

    except Exception as e:
        messagebox.showerror("Error", f"Error al procesar la imagen: {e}")


# Crear la ventana principal
root = tk.Tk()
root.title("Test Comparación Facial")

# Configurar tamaño de ventana
root.geometry("400x200")

# Título
title_label = tk.Label(root, text="Comparador de Rostros", font=("Helvetica", 16))
title_label.pack(pady=20)

# Botón para seleccionar imagen y comparar
compare_button = tk.Button(root, text="Seleccionar imagen y comparar", command=select_and_compare,
                           font=("Helvetica", 12))
compare_button.pack(pady=20)

# Iniciar loop de la GUI
root.mainloop()
