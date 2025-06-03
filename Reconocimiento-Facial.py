import pandas as pd
import os
from IPython.display import Image, display

# Ruta a tu archivo Excel (cámbiala si usas otro nombre o ruta)
excel_path = 'dataset.xlsx'  # Cambia a tu archivo .xlsx o .csv

# Para probar, supondré que es un archivo Excel:
# Si es CSV, usa pd.read_csv()

# Cargar Excel (en tu caso deberás subirlo o usar ruta válida)
df = pd.read_excel('dataset.xlsx')  # Cambia por tu ruta en Colab

# Validar si archivo de imagen existe y mostrarla
def mostrar_imagen(ruta_imagen):
    if isinstance(ruta_imagen, str) and os.path.isfile(ruta_imagen):
        print(f"Imagen encontrada: {ruta_imagen}")
        display(Image(filename=ruta_imagen))
    else:
        print(f"Imagen NO encontrada o ruta inválida: {ruta_imagen}")

# Recorremos el DataFrame para validar la columna Foto
for idx, row in df.iterrows():
    print(f"Procesando persona: {row['Nombre']} {row['Apellido']}")
    mostrar_imagen(row['Foto'])

# Añadimos la columna Kp con diccionario vacío como placeholder
df['Kp'] = [{} for _ in range(len(df))]

# Mostrar DataFrame actualizado
print("\nDataFrame actualizado:")
print(df)
