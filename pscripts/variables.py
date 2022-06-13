import tkinter as tk


class ValueProperty:
    @property
    def value(self):
        return self.get()

    @value.setter
    def value(self, value):
        self.set(value)


class StringVar(tk.StringVar, ValueProperty):
    pass


class IntVar(tk.IntVar, ValueProperty):
    pass


class BooleanVar(tk.BooleanVar, ValueProperty):
    pass
