'''
gui for 1d simulator 
'''

import tkinter as tk  # Official tkinter package

class GUI(tk.Tk):
    def __init__(self, parent):
        tk.Tk.__init__(self, parent)
        self.parent = parent

        self.time = None
        self.velocity = None
        self.axes = None

        #self = tk.Tk()
        self.title("1D Simulation Parameters")
        self.geometry("400x300")

        label = tk.Label(self, text="Select Parameters for Simulation")
        label.pack(pady=10)
        ############axes 
        axes_label = tk.Label(self, text="Select which axes you'd like to visualize:")
        axes_label.pack(pady=5)
        axes_options = ["x", "y", "z"]
        self.axes_var = tk.StringVar(self)  
        self.axes_var.set(axes_options[0])  # Default selection (x)

        axes_menu = tk.OptionMenu(self, self.axes_var, *axes_options)
        axes_menu.pack(pady=5)

        ##############time 
        time_label = tk.Label(self, text="Select Time in seconds (minimum: 50):")
        time_label.pack(pady=5)
        self.time_entry = tk.Entry(self)
        self.time_entry.pack(pady=5)
        
        ############velocity 
        velocity_label = tk.Label(self, text="Select Velocity:")
        velocity_label.pack(pady=5)
        self.velocity_entry = tk.Entry(self)
        self.velocity_entry.pack(pady=5)

        run_button = tk.Button(self, text="Run Simulation", command=self.getVals)
        run_button.pack(pady=20)

        # self.mainloop()

    def getVals(self):
        self.time = self.time_entry.get()
        if self.time == "":
            Win2 = tk.Tk()
            Win2.withdraw()

        self.velocity = self.velocity_entry.get()
        if self.velocity == "":
            Win2 = tk.Tk()
            Win2.withdraw()

        self.axes = self.axes_var.get()
        if self.axes == "":
            Win2 = tk.Tk()
            Win2.withdraw()

        self.quit()
