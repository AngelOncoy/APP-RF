# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label

from plyer import filechooser  # Para abrir la galería
from plyer import camera  # Para tomar foto con cámara

import Reconocimiento_Facial  # Tu módulo de reconocimiento facial

import os

class MainApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')

        # Imagen a mostrar
        self.img = Image()
        self.layout.add_widget(self.img)

        # Label para mostrar resultados
        self.lbl_resultado = Label(text='Resultado: ', size_hint_y=None, height=50)
        self.layout.add_widget(self.lbl_resultado)

        # Botón para abrir galería
        btn_load = Button(text='Seleccionar imagen de galería')
        btn_load.bind(on_press=self.open_gallery)
        self.layout.add_widget(btn_load)

        # Botón para tomar foto con cámara
        btn_camera = Button(text='Tomar foto con cámara')
        btn_camera.bind(on_press=self.tomar_foto)
        self.layout.add_widget(btn_camera)

        return self.layout

    # Función que abre la galería usando plyer
    def open_gallery(self, instance):
        filechooser.open_file(on_selection=self.selected, filters=['*.jpg', '*.jpeg', '*.png'])

    # Función que recibe la imagen seleccionada
    def selected(self, selection):
        if selection:
            imagen_path = selection[0]

            # Mostrar imagen en la app
            self.img.source = imagen_path
            self.img.reload()

            # Llamar a tu función de reconocimiento facial
            persona, resultado = Reconocimiento_Facial.buscar_persona_por_imagen(imagen_path)

            # Mostrar resultado en label
            if persona is None:
                texto_resultado = f"No encontrado. {resultado}"
            else:
                texto_resultado = (f"✅ Encontrado: {persona['nombre']} {persona['apellido']} "
                                   f"({persona['correo']})\nSimilitud: {resultado*100:.2f}%")

            self.lbl_resultado.text = texto_resultado

    # Función que toma foto con cámara
    def tomar_foto(self, instance):
        # Definimos ruta temporal para la foto
        output_path = os.path.join(os.getcwd(), 'foto_temp.jpg')
        camera.take_picture(output_path, self.foto_tomada)

    # Callback cuando se toma la foto
    def foto_tomada(self, path_foto):
        if path_foto:
            # Mostrar imagen en la app
            self.img.source = path_foto
            self.img.reload()

            # Llamar a tu función de reconocimiento facial
            persona, resultado = Reconocimiento_Facial.buscar_persona_por_imagen(path_foto)

            # Mostrar resultado en label
            if persona is None:
                texto_resultado = f"No encontrado. {resultado}"
            else:
                texto_resultado = (f"✅ Encontrado: {persona['nombre']} {persona['apellido']} "
                                   f"({persona['correo']})\nSimilitud: {resultado*100:.2f}%")

            self.lbl_resultado.text = texto_resultado

if __name__ == '__main__':
    MainApp().run()
