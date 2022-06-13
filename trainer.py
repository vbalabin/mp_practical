import os
import cv2
import numpy as np
import random as rd
import tkinter as tk
from threading import Thread
from pscripts.neural import NeuralNetwork
from pscripts.app import TkApp
from pscripts.variables import IntVar, StringVar, BooleanVar
from pscripts.constants import NN_LAYERS_SIZES, LEARNING_RATE


class NetworkTrainer(TkApp):

    APP_FONT = ('Consolas', '16', 'bold')

    def __init__(self, master):
        self.master = master
        self.grid_ = self.add_grid(master, 11, 7)

        self.variable_nn_path = StringVar()
        self.variable_nn_path.value = os.getcwd() + '\\neuronet\\nn-current.pickle'

        self.variable_train_path = StringVar(value=os.getcwd()+'\\train')

        self.variable_samples = IntVar(value=60000)

        self.variable_epoch = IntVar()
        self.variable_correct = IntVar()
        self.variable_errors = IntVar()

        self.variable_status_bar = StringVar(value="..")

        self.variable_recreate_nn = BooleanVar(value=False)

        self.isbusy = False
        self.configure_main_window()
        self.setup_ui()

    def setup_ui(self):
        # 1 line
        r = 0
        self.label_nn_path = tk.Label(
            self.master, text='путь к нейросети: ', font=self.APP_FONT)
        self.label_nn_path.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.entry_nn_path = tk.Entry(
            self.master, textvariable=self.variable_nn_path, width=55)
        self.entry_nn_path.grid(row=r, column=2, columnspan=4, sticky=tk.W)

        # 2 line
        r = 1
        self.label_train_path = tk.Label(
            self.master, text='путь к выборке: ', font=self.APP_FONT)
        self.label_train_path.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.entry_train_path = tk.Entry(
            self.master, textvariable=self.variable_train_path, width=55)
        self.entry_train_path.grid(row=r, column=2, columnspan=7, sticky=tk.W)

        # 3 line
        r = 2
        self.label_samples = tk.Label(
            self.master, text='файлов в выборке: ', font=self.APP_FONT)
        self.label_samples.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.entry_samples = tk.Entry(
            self.master, textvariable=self.variable_samples, width=55)
        self.entry_samples.grid(row=r, column=2, columnspan=7, sticky=tk.W)

        # 4 line
        r = 3
        self.label_new = tk.Label(
            self.master, text='Пересоздать сеть: ', font=self.APP_FONT)
        self.label_new.grid(row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.checkbox_resize = tk.Checkbutton(
            self.master, variable=self.variable_recreate_nn)
        self.checkbox_resize.grid(
            row=r, column=2, columnspan=1, sticky=tk.W, padx=8)

        # 6 line
        r = 5
        self.label_epoch = tk.Label(
            self.master, text='Эпоха: ', font=self.APP_FONT)
        self.label_epoch.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.label_epoch_value = tk.Label(
            self.master, textvariable=self.variable_epoch, text='----', font=self.APP_FONT)
        self.label_epoch_value.grid(row=r, column=3, columnspan=1, sticky=tk.E)

        # 7 line
        r = 6
        self.label_error_sum = tk.Label(
            self.master, text='Правильно из 100:', font=self.APP_FONT)
        self.label_error_sum.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.label_epoch_value = tk.Label(
            self.master, textvariable=self.variable_correct, font=self.APP_FONT)
        self.label_epoch_value.grid(row=r, column=3, columnspan=1, sticky=tk.E)

        # 8 line
        r = 7
        self.label_error_sum = tk.Label(
            self.master, text='Сумма ошибок: ', font=self.APP_FONT)
        self.label_error_sum.grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8)
        self.label_epoch_value = tk.Label(
            self.master, textvariable=self.variable_errors, font=self.APP_FONT)
        self.label_epoch_value.grid(row=r, column=3, columnspan=1, sticky=tk.E)

        # 10 line
        r = 9
        self.button_run = tk.Button(
            self.master, text='Обучать', width=15, command=self.__train_btn_on_click)
        self.button_run.grid(row=r, column=2, columnspan=2, pady=5)

        # 11 line
        r = 10
        self.label_status_bar = tk.Label(
            self.master, textvariable=self.variable_status_bar)
        self.label_status_bar.grid(
            row=r, column=0, columnspan=7, sticky=tk.W, padx=8)

    def __train_btn_on_click(self):
        """
        """
        self.set_main_button_disabled()
        self.load_neural_network()
        self.load_dataset()

    def load_neural_network(self):
        """
        """
        isloaded = False
        if self.variable_recreate_nn.value:
            self.neural_network = NeuralNetwork(LEARNING_RATE, NN_LAYERS_SIZES)
            try:
                self.save_nnetwork_to_hdd(
                    self.neural_network, self.variable_nn_path.value)
                isloaded = True
            except:
                pass
        else:
            try:
                self.neural_network: NeuralNetwork = self.load_nnetwork_from_hdd(
                    self.variable_nn_path.value)
                isloaded = True
            except:
                pass

        if not isloaded:
            self.variable_status_bar.value = "С файлом нейронной сети что-то не так"
        else:
            self.set_main_button_enabled()

    def set_main_button_disabled(self):
        """
        """
        self.button_run["state"] = "disabled"
        self.isbusy = True

    def set_main_button_enabled(self):
        """
        """
        self.button_run["state"] = "normal"
        self.isbusy = False

    def load_dataset(self):
        """
        """
        if self.isbusy:
            self.set_main_button_enabled()
        else:
            self.set_main_button_disabled()
            self.variable_status_bar.value = "Загружаю датасет .."
            data_thread = Thread(target=self.load_data_set_job, daemon=True)
            data_thread.start()

    def load_data_set_job(self):
        """
        """
        samples = self.variable_samples.value
        datapath = self.variable_train_path.value

        self.inputs = np.zeros((784,))
        np.set_printoptions(suppress=True)

        for dirpath, _, filenames in os.walk(datapath):
            break

        image_filenames = filenames[:samples]
        self.inputs = np.empty(shape=(samples, 784), dtype='object')
        self.digits = np.empty(shape=(samples,), dtype=np.int8)
        for i in range(len(image_filenames)):
            _image = cv2.imread(f"{dirpath}/{image_filenames[i]}", 0)
            self.inputs[i] = _image.ravel() / 255.0
            self.digits[i] = int(image_filenames[i][-5])

        self.variable_status_bar.value = "Датасет загружен"

        self.set_main_button_enabled()
        self.button_run.configure(
            text="Остановить", command=self.__changed_btn_on_click)
        self.is_training_job_needed = True
        self.train_cycle_thread = Thread(
            target=self.train_cycle_job, daemon=True)
        self.train_cycle_thread.start()

    def __changed_btn_on_click(self):
        """
        """
        self.is_training_job_needed = False
        self.variable_status_bar.value = "Жду конца эпохи, чтобы сохранить .."
        self.set_main_button_disabled()

    def train_cycle_job(self):
        """
        """
        samples = self.variable_samples.value
        nnetwork = self.neural_network

        while self.is_training_job_needed:
            self.variable_status_bar.value = "Обучаю сеть .."
            right = 0
            error_sum = 0
            batchSize = 100
            nnetwork.epoch += 1
            self.variable_epoch.value = nnetwork.epoch

            for j in range(batchSize):

                if j % 5 == 0:
                    self.variable_correct.value = right
                    self.variable_errors.value = int(error_sum)

                imgIndex = int(rd.random() * samples)
                targets = np.zeros((10,))
                digit = self.digits[imgIndex]
                targets[digit] = 1

                outputs = nnetwork.feed_forfard(self.inputs[imgIndex])
                max_digit = 0
                max_digit_weight = -1
                for k in range(10):
                    if(outputs[k] > max_digit_weight):
                        max_digit_weight = outputs[k]
                        max_digit = k
                if(digit == max_digit):
                    right += 1
                for k in range(10):
                    error_sum += (targets[k] - outputs[k]) * \
                        (targets[k] - outputs[k])
                nnetwork.back_propagation(targets)

        else:
            self.save_nnetwork_to_hdd(
                self.neural_network, self.variable_nn_path.value)
            self.button_run.configure(
                text='Обучать', command=self.__train_btn_on_click)
            self.set_main_button_enabled()
            self.variable_status_bar.value = ".."

    def configure_main_window(self):
        """
        main window properties
        """
        self.master.title('Trainer')
        self.master.resizable(False, False)


if __name__ == "__main__":
    root = tk.Tk()
    im = NetworkTrainer(root)
    root.mainloop()
