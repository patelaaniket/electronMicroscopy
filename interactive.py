print("Loading modules...")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pixstem.api as ps
import multiprocessing
import hyperspy.api as hs
import ctypes
import csv
import tkinter as tk
from tkinter import font
print("Modules loaded.")

file = None
distances = None
currFunc = None

def getEntry(event):
    if currFunc is "loadFile":
        loadFile(entry.get())
    elif currFunc is "analysis":
        analysis(entry.get())
    elif currFunc is "toCSV":
        toCSV(entry.get())
    

def loadFile(filename = None):
    global currFunc
    global file
    currFunc = "loadFile"

    if filename is None or filename is "":
        entry.bind("<Return>", getEntry)
        label1['text'] = label1['text'] + "Please enter the path of the input file in the text box provided then press Enter.\n"
    else:
        label1['text'] = label1['text'] + "Loading file...\n"
        file = hs.load(filename)
        label1['text'] = label1['text'] + "File loaded.\n"
        entry.delete(0, tk.END)
        entry.unbind("<Return>")

def distance(x1, y1, x2, y2):
    return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

def pos_dist(x1, y1, x2, y2):
    if (x2 > x1 and y2 > y1):
        return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    else:
        return -1.0

def findCenter(im, peak):
    center = (0,0)
    maximum = 0
    for (x,y) in np.ndenumerate(peak):
        for (a, b) in y:
            if (int(a) < len(im) and int(b) < len(im) and im[int(a)][int(b)] > maximum):
                maximum = im[int(a)][int(b)]
                center = (b, a)
    return center

def multiprocessing_func(i, j, rnd):
    s = ps.PixelatedSTEM(hs.signals.Signal2D(file.inav[i, j]))
    imarray = np.array(s)
    s = s.rotate_diffraction(0,show_progressbar=False)
    ############################################################################################################################
    st = s.template_match_disk(disk_r=5, lazy_result=False, show_progressbar=False)
    peak_array = st.find_peaks(lazy_result=False, show_progressbar=False)
    peak_array_com = s.peak_position_refinement_com(peak_array, lazy_result=False, show_progressbar=False)
    s_rem = s.subtract_diffraction_background(lazy_result=False, show_progressbar=False)
    peak_array_rem_com = s_rem.peak_position_refinement_com(peak_array, lazy_result=False, show_progressbar=False)
    ############################################################################################################################
    center = findCenter(imarray, peak_array_rem_com)

    # finds the specific spot and adding that distance to the array
    posDistance = 0
        
    for (x,y) in np.ndenumerate(peak_array_rem_com):
        prev = (0, 0)
        for (a, b) in y:
            if abs(center[0] - b) < 1E-5 and abs(center[1] - a) < 1E-5:
                posDistance = distance(center[0], center[1], prev[1], prev[0])
                break
            prev = (a, b)
    distances[j][i] = round(posDistance, rnd)

def analysis(values = None):
    global file, currFunc, distances
    currFunc = "analysis"

    if file is None:
        label1['text'] = label1['text'] + "Please load a file before starting analysis.\n"
    elif values is None:
        entry.bind("<Return>", getEntry)
        label1['text'] = label1['text'] + "Please enter the number of rows and columns you would like to analyze and the number of decimal point to which to round values to seperated by spaces. All values are integers\n"
    else:
        label1['text'] = label1['text'] + "Starting analysis...\n"
        t = values.split(" ")
        COL = int(t[1])
        rnd = int(t[2])
        ROW = int(t[0])

        shared_array_base = multiprocessing.Array(ctypes.c_double, ROW*COL)
        distances = np.ctypeslib.as_array(shared_array_base.get_obj())
        distances = distances.reshape(COL, ROW)

        for i in range(ROW):
            print(i)
            processes = []
            for j in range(COL):
                p = multiprocessing.Process(target=multiprocessing_func, args=(i, j, rnd,))
                processes.append(p)
                p.start()

            for process in processes:
                process.join()  
        label1['text'] = label1['text'] + "Analysis complete.\n"
        entry.delete(0, tk.END)
        entry.unbind("<Return>")

def toCSV(filename = None):
    global currFunc
    currFunc = "toCSV"
    if distances is None:
        label1['text'] = label1['text'] + "Please analyze a file before saving data.\n"
    elif filename is None or filename is "":
        entry.bind("<Return>", getEntry)
        label1['text'] = label1['text'] + "Please enter the path of the file you want to save to in the text box provided then press Enter.\n"
    else:
        file = open(filename, "w")
        writer = csv.writer(file)
        for i in distances:
            writer.writerow(i)
        file.close()
        label1['text'] = label1['text'] + "File saved.\n"
        entry.delete(0, tk.END)
        entry.unbind("<Return>")

def barChart(INTERVAL = 0.01):
    global distances
    if distances is None:
        label1['text'] = label1['text'] + "Please analyze a file before saving data.\n"
    else:
        dist = distances.copy()
        dist = dist.flatten()
        x_pos = np.arange(np.min(dist), np.max(dist), INTERVAL) # this 0.01 is the distance between each x-axis label. So for example it goes 1.0, 1.01, 1.02, 1.03...
        x_pos = [round(num, 2) for num in x_pos]
        y_pos = np.arange(len(x_pos))
        ################################################################################################################################
        from collections import Counter
        counter = Counter(dist)
        counts = []
        for i in x_pos:
                counts.append(counter[i]) if i in counter.keys() else counts.append(0)
        ################################################################################################################################
        plt.bar(y_pos, counts, align='center', alpha=0.95) # creates the bar plot
        plt.xticks(y_pos, x_pos, fontsize = 5)
        plt.xlabel('Distance from center peek', fontsize = 5)
        plt.ylabel('Counts', fontsize = 5)
        plt.title('Distance Counts', fontsize = 5)

        ax = plt.gca()
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.tick_params(axis='x', which='major', labelsize=5)
        ax.tick_params(axis='y', which='major', labelsize=5)

        [l.set_visible(False) for (i,l) in enumerate(ax.xaxis.get_ticklabels()) if i % 2 != 0] 
        # The '2' is the every nth number of labels its shows on the x-axis. So rn is shows every 2nd label. 

        plt.gcf().subplots_adjust(bottom = 0.23)
        plt.rcParams["figure.dpi"] = 100
        plt.savefig("temporary.png")
        
        r = tk.Tk()
        c = tk.Canvas(r, height=720, width=1080)
        c.pack()

        img = tk.PhotoImage(master=c, file='temporary.png')
        backLabel = tk.Label(r, image=img)
        backLabel.place(relx=0, rely=0, relwidth=1, relheight=1)

        r.mainloop()

def heatMap():
    global distances
    if distances is None:
        label1['text'] = label1['text'] + "Please analyze a file before saving data.\n"
    else:
        data = distances.copy()
        df = pd.DataFrame(data, columns=np.arange(len(data[0])), index=np.arange(len(data)))
        from matplotlib import cm as cm
        fig, a = plt.subplots(figsize=(6,5.5)) 
        yeet = sns.heatmap(df, cmap=cm.get_cmap("RdYlBu_r"),ax=a)
        fig = yeet.get_figure()
        fig.savefig("temporary.png")

        r = tk.Tk()
        c = tk.Canvas(r, height=720, width=1080)
        c.pack()

        img = tk.PhotoImage(master=c, file='temporary.png')
        backLabel = tk.Label(r, image=img)
        backLabel.place(relx=0, rely=0, relwidth=1, relheight=1)

        r.mainloop()


HEIGHT = 900
WIDTH = 1400

root = tk.Tk()

canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()
frame = tk.Frame(root, bg='#333333')
frame.place(relwidth=1, relheight=1)

# Menu Label
label = tk.Label(frame, text='Menu', bg='#333333', font=('Times New Roman', 50), fg='#ffffff')
label.place(relx=0.45, rely=0.05, relwidth=0.1, relheight=0.05)

# Text Output box
label1 = tk.Message(frame, bg='#999999', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness = 0, bd=0, width = 1100)
label1.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.35)

# Entry box
entry = tk.Entry(frame, font=40)
entry.place(relx=0.1, rely=0.9, relwidth=0.8, relheight=0.05)

# Buttons
button = tk.Button(frame, text='Load File', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: loadFile(), pady=0.02, fg='#ffffff')
button.place(relx=0.42, rely=0.15, relwidth=0.16, relheight=0.05)

button1 = tk.Button(frame, text='Start Analysis', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: analysis(), pady=0.02, fg='#ffffff')
button1.place(relx=0.39, rely=0.22, relwidth=0.22, relheight=0.05)

button2 = tk.Button(frame, text='Create Bar Chart', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: barChart(), pady=0.02, fg='#ffffff')
button2.place(relx=0.375, rely=0.29, relwidth=0.25, relheight=0.05)

button3 = tk.Button(frame, text='Create Heat Map', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: heatMap(), pady=0.02, fg='#ffffff')
button3.place(relx=0.38, rely=0.36, relwidth=0.24, relheight=0.05)

button4 = tk.Button(frame, text='Transfer Data to .csv', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: toCSV(), pady=0.02, fg='#ffffff')
button4.place(relx=0.34, rely=0.43, relwidth=0.32, relheight=0.05)

root.mainloop()