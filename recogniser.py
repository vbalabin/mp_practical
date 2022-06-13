import os
import numpy as np
import tkinter as tk
from PIL import ImageGrab, Image
from threading import Thread
from pscripts.app import TkApp
from pscripts.neural import NeuralNetwork
from pscripts.variables import IntVar, StringVar


class NetworkRecogniser(TkApp):

    APP_FONT = ('Consolas', '16', 'bold')
    BIG_FONT = ('Consolas', '36', 'bold')

    def __init__(self, master):
        self.master = master
        self.grid_ = self.add_grid(master, 17, 10)

        self.neural_network = None
        self.variable_nn_path = StringVar()
        self.variable_nn_path.value = os.getcwd() + '\\neuronet\\nn-current.pickle'

        self.variable_answer = IntVar(value=0)

        self.variable_status_bar = StringVar(value="Загрузить нейросеть")

        self.configure_main_window()
        self.setup_ui()
        self.prepare_network()

    def setup_ui(self):
        # 1 line
        r = 0
        self.label_nn_path = tk.Label(
            self.master, text='путь к нейросети: ', font=self.APP_FONT)
        self.label_nn_path.grid(
            row=r, column=0, columnspan=4, sticky=tk.W, padx=8)
        self.entry_nn_path = tk.Entry(
            self.master, textvariable=self.variable_nn_path, width=55)
        self.entry_nn_path.grid(row=r, column=4, columnspan=5, sticky=tk.W)

        # 2 line
        r = 1
        self.button_run = tk.Button(self.master, text='Загрузить', width=15,
                                    command=self.prepare_network)
        self.button_run.grid(row=r, column=0, columnspan=2, pady=5)

        # 4 line
        r, c = 3, 2
        self.canvas = tk.Canvas(self.master, bg='black', width=280, height=280)
        self.canvas.grid(row=r, column=c, columnspan=6, rowspan=10)
        self.old_x, self.old_y = None, None
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset)
        self.canvas.bind('<ButtonRelease-3>', self.clear)

        # 15 line
        r = 14
        self.label_answer = tk.Label(
            self.master, text='Ответ:', font=self.BIG_FONT)
        self.label_answer.grid(
            row=r, column=3, columnspan=3, sticky=tk.W, padx=8)
        self.label_answer_value = tk.Label(
            self.master, textvariable=self.variable_answer, font=self.BIG_FONT)
        self.label_answer_value.grid(
            row=r, column=6, columnspan=1, sticky=tk.W, padx=8)

        # 17 line
        r = 16
        self.label_status_bar = tk.Label(
            self.master, textvariable=self.variable_status_bar)
        self.label_status_bar.grid(
            row=r, column=0, columnspan=7, sticky=tk.W, padx=8)

    def paint(self, event):
        line_width = 16
        paint_color = 'white'
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y,
                                    width=line_width, fill=paint_color,
                                    capstyle='round', smooth=False, splinesteps=32)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None
        self.thread = Thread(target=self.recognise_image, daemon=True)
        self.thread.start()

    def clear(self, event):
        self.canvas.delete("all")

    def prepare_network(self):
        """
        """
        try:
            self.neural_network: NeuralNetwork = self.load_nnetwork_from_hdd(
                self.variable_nn_path.value)
            self.variable_status_bar.value = "Сеть загружена"
        except:
            self.variable_status_bar.value = "С путем к файлу сети что-то не так"
            raise

    def get_image_from_canvas(self):
        left_x = self.canvas.winfo_rootx()
        left_y = self.canvas.winfo_rooty()
        right_x = self.canvas.winfo_rootx() + self.canvas.winfo_width()
        right_y = self.canvas.winfo_rooty() + self.canvas.winfo_height()
        image = ImageGrab.grab(
            bbox=(left_x, left_y, right_x, right_y), all_screens=True)
        self.image = image.resize((28, 28), Image.ANTIALIAS)
        self.image = self.image.convert('L')
        self.image.save('gray.png')

    def call_network(self):
        _image = np.array(self.image)
        input = _image.ravel() / 255.0
        ans = self.neural_network.feed_forfard(input)
        self.variable_answer.value = np.argmax(ans)
        print(ans)
        print('-'*48)

    def recognise_image(self):
        self.get_image_from_canvas()
        self.call_network()

    def configure_main_window(self):
        """
        main window properties
        """
        self.master.title('Recogniser')
        self.master.resizable(False, False)


if __name__ == "__main__":
    root = tk.Tk()
    im = NetworkRecogniser(root)
    root.mainloop()
