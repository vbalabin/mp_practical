import pickle
import tkinter as tk


class TkApp:

    def save_nnetwork_to_hdd(self, nn, nnpath):
        with open(nnpath, 'wb') as handle:
            pickle.dump(nn, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_nnetwork_from_hdd(self, nnpath):
        with open(nnpath, 'rb') as handle:
            return pickle.load(handle)

    def add_grid(self, master, sizex=8, sizey=8):
        """
        creates a table structure on a given frame
        """
        result = list()
        for i in range(sizex):
            result.append(list())
            tk.Grid.rowconfigure(master, i, weight=0)
            for j in range(sizey):
                frame = tk.Frame(master, width=50, height=30)
                frame.grid(row=i, column=j, sticky=tk.NSEW)
                tk.Grid.columnconfigure(master, j, weight=0)
                result[i].append(frame)
        return result
