#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.predict_gui import BasePredictGUI
from src.predict import Predict
from PIL import Image
from DeepLearningTemplate.data_preparation import parse_transforms
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import messagebox
import gradio as gr
import tkinter as tk


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                                     '.pgm', '.tif', '.tiff', '.webp'))
        self.predictor = Predict(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        self.loader = Image.open
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.color_space = project_parameters.color_space
        self.web_interface = project_parameters.web_interface
        self.examples = project_parameters.examples if len(
            project_parameters.examples) else None

        # matplotlib canvas
        # this is Tkinter default background-color
        facecolor = (0.9254760742, 0.9254760742, 0.9254760742)
        figsize = np.array([12, 4])
        self.image_canvas = FigureCanvasTkAgg(Figure(figsize=figsize,
                                                     facecolor=facecolor),
                                              master=self.window)

        # button
        self.generate_button = self.recognize_button
        self.generate_button.config(text='Generate', command=self.generate)

    def reset_widget(self):
        super().reset_widget()
        self.image_canvas.figure.clear()

    def resize_image(self, image):
        width, height = image.size
        ratio = max(self.window.winfo_height(),
                    self.window.winfo_width()) / max(width, height)
        ratio *= 0.25
        return image.resize((int(width * ratio), int(height * ratio)))

    def display(self):
        image = self.loader(self.filepath).convert(self.color_space)
        resized_image = self.resize_image(image=image)
        # set the cmap as gray if resized_image doesn't exist channel
        cmap = 'gray' if len(np.array(resized_image).shape) == 2 else None
        rows, cols = 1, 1
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            subplot.imshow(resized_image, cmap=cmap)
            subplot.axis('off')
        self.image_canvas.draw()

    def open_file(self):
        super().open_file()
        self.display()

    def display_output(self, generated_sample):
        self.image_canvas.figure.clear()
        sample = self.loader(fp=self.filepath).convert(self.color_space)
        sample = self.transform(sample)
        # the sample dimension is (in_chans, width, height)
        sample = sample.cpu().data.numpy()
        # the generated_sample dimension is (1, in_chans, width, height),
        # so use 0 index to get the first generated_sample
        generated_sample = generated_sample[0]
        if sample.shape[0] == 1:
            # delete channels axis, so the dimension is (width, height)
            cmap = 'gray'
            sample = sample[0]
            generated_sample = generated_sample[0]
        else:
            # transpose the dimension to (width, height, in_chans)
            cmap = None
            sample = sample.transpose(1, 2, 0)
            generated_sample = generated_sample.transpose(1, 2, 0)
        rows, cols = 1, 2
        title = ['original', 'generated']
        for idx in range(1, rows * cols + 1):
            subplot = self.image_canvas.figure.add_subplot(rows, cols, idx)
            subplot.title.set_text('{}'.format(title[(idx - 1) % 3]))
            if (idx - 1) % 3 == 0:
                # plot real
                subplot.imshow(sample, cmap=cmap)
            elif (idx - 1) % 3 == 1:
                # plot fake
                subplot.imshow(generated_sample, cmap=cmap)
            subplot.axis('off')
        self.image_canvas.draw()

    def generate(self):
        if self.filepath is not None:
            generated_sample = self.predictor.predict(inputs=self.filepath)
            self.display_output(generated_sample=generated_sample)
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def inference(self, inputs):
        # the dimension of generated_sample is (batch, channels, height, width)
        generated_sample = self.predictor.predict(inputs=inputs)
        batch, channels, height, width = generated_sample.shape
        if channels == 1:
            generated_sample = generated_sample[0, 0]
        else:
            generated_sample = generated_sample[0].transpose(1, 2, 0)
        return generated_sample[:, :, 0]

    def run(self):
        if self.web_interface:
            gr.Interface(fn=self.inference,
                         inputs=gr.inputs.Image(image_mode=self.color_space,
                                                type='filepath'),
                         outputs=gr.outputs.Image(type='numpy'),
                         examples=self.examples,
                         interpretation="default").launch(share=True,
                                                          inbrowser=True)
        else:
            # NW
            self.open_file_button.pack(anchor=tk.NW)
            self.recognize_button.pack(anchor=tk.NW)

            # N
            self.filepath_label.pack(anchor=tk.N)
            self.image_canvas.get_tk_widget().pack(anchor=tk.N)
            self.predicted_label.pack(anchor=tk.N)
            self.result_label.pack(anchor=tk.N)

            # run
            super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
