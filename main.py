import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.camera import Camera
from kivy.uix.filechooser import FileChooserIconView
import io

class SpectroscopyApp(App):
    def build(self):
        self.title = 'Mobile Spectroscopy App'
        self.reference_image_rgb = None
        self.sample_image_rgb = None
        
        layout = BoxLayout(orientation='vertical')
        
        button_layout = BoxLayout(size_hint_y=None, height=50)
        
        self.reference_button = Button(text='Capture Reference Spectrum Image')
        self.reference_button.bind(on_release=self.capture_reference_image)
        button_layout.add_widget(self.reference_button)
        
        self.sample_button = Button(text='Capture Sample Spectrum Image', disabled=True)
        self.sample_button.bind(on_release=self.capture_sample_image)
        button_layout.add_widget(self.sample_button)
        
        self.analyze_button = Button(text="Analyze Spectrum", disabled=True)
        self.analyze_button.bind(on_release=self.analyze_spectrum)
        button_layout.add_widget(self.analyze_button)
        
        self.save_button = Button(text="Save Spectrum", disabled=True)
        self.save_button.bind(on_release=self.save_spectrum)
        button_layout.add_widget(self.save_button)
        
        layout.add_widget(button_layout)
        
        self.image_display = Image()
        layout.add_widget(self.image_display)
        
        self.absorbance_plot = Image()
        layout.add_widget(self.absorbance_plot)
        
        return layout

    def capture_reference_image(self, instance):
        self.open_camera(self.load_reference_image)

    def capture_sample_image(self, instance):
        self.open_camera(self.load_sample_image)

    def open_camera(self, load_image_callback):
        content = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True)
        content.add_widget(self.camera)

        popup = Popup(title="Capture Spectrum Image", content=content, size_hint=(0.9, 0.9))
        capture_button = Button(text="Capture", size_hint=(1, 0.1), on_release=lambda x: self.on_capture_button_pressed(load_image_callback, popup))
        content.add_widget(capture_button)
        content.add_widget(Button(text="Cancel", size_hint=(1, 0.1), on_release=popup.dismiss))
        popup.open()

    def on_capture_button_pressed(self, load_image_callback, popup):
        self.camera.export_to_png("captured_image.png")
        load_image_callback("captured_image.png")
        popup.dismiss()

    def load_reference_image(self, path):
        self.reference_image_rgb = self.load_image(path)
        self.reference_spectrum = self.extract_spectrum(self.reference_image_rgb)
        self.reference_colors = [color for wl, color in self.reference_spectrum]
        self.tree = KDTree(self.reference_colors)
        self.sample_button.disabled = False
        texture = self.cv2_to_texture(self.reference_image_rgb)
        self.image_display.texture = texture

    def load_sample_image(self, path):
        self.sample_image_rgb = self.load_image(path)
        self.analyze_button.disabled = False
        texture = self.cv2_to_texture(self.sample_image_rgb)
        self.image_display.texture = texture

    def load_image(self, path):
        image = cv2.imread(path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb

    def cv2_to_texture(self, image_rgb):
        buf = cv2.flip(image_rgb, 0).tobytes()
        texture = Texture.create(size=(image_rgb.shape[1], image_rgb.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def extract_spectrum(self, image_rgb):
        height, width, _ = image_rgb.shape
        wavelengths = np.linspace(400, 700, width)  # Approximate visible spectrum range in nm
        spectrum = []

        for x in range(width):
            avg_color = np.mean(image_rgb[:, x, :], axis=0)
            spectrum.append((wavelengths[x], avg_color))

        return spectrum

    def get_wavelength_from_color(self, r, g, b):
        distance, index = self.tree.query([r, g, b])
        wavelength = self.reference_spectrum[index][0]
        return wavelength

    def get_RGB_and_intensity(self, image_rgb):
        height, width, _ = image_rgb.shape
        wavelengths = []
        intensities = []

        for x in range(width):
            avg_color = np.mean(image_rgb[:, x, :], axis=0)
            r, g, b = avg_color
            wavelength = self.get_wavelength_from_color(r, g, b)
            intensity = 0.2126 * r + 0.7152 * g + 0.0722 * b
            wavelengths.append(wavelength)
            intensities.append(intensity)

        return wavelengths, intensities

    def analyze_spectrum(self, instance):
        reference_wavelengths, reference_intensities = self.get_RGB_and_intensity(self.reference_image_rgb)
        sample_wavelengths, sample_intensities = self.get_RGB_and_intensity(self.sample_image_rgb)

        min_length = min(len(reference_intensities), len(sample_intensities))
        absorbance = []

        for i in range(min_length):
            if sample_intensities[i] == 0 or reference_intensities[i] == 0:
                absorbance.append(0)
            else:
                absorbance.append(np.log10(reference_intensities[i] / sample_intensities[i]))

        fig, ax = plt.subplots()
        ax.plot(reference_wavelengths[:min_length], absorbance, marker='o', linestyle='-', markersize=1, linewidth=0.2)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Absorbance')
        ax.set_title('Absorbance vs Wavelength')
        ax.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), 1)
        plot_image_rgb = cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB)
        texture = self.cv2_to_texture(plot_image_rgb)
        self.absorbance_plot.texture = texture
        
        self.absorbance_plot_buffer = buf
        self.save_button.disabled = False

    def save_spectrum(self, instance):
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserIconView()
        content.add_widget(filechooser)
        save_button = Button(text="Save", size_hint=(1, 0.1))
        save_button.bind(on_release=lambda x: self.save_to_file(filechooser.path, filechooser.selection))
        content.add_widget(save_button)
        content.add_widget(Button(text="Cancel", size_hint=(1, 0.1), on_release=self.dismiss_popup))

        self.popup = Popup(title="Save Spectrum", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def save_to_file(self, path, selection):
        if selection:
            save_path = selection[0]
        else:
            save_path = path + "/spectrum.png"

        with open(save_path, 'wb') as f:
            f.write(self.absorbance_plot_buffer.getbuffer())

        self.popup.dismiss()

    def dismiss_popup(self, instance):
        self.popup.dismiss()

if __name__ == '__main__':
    SpectroscopyApp().run()
