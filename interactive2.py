print("Loading modules...")
from os import remove, environ, path
environ["DISPLAY"] = ":0"
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pixstem.api as ps
import multiprocessing
import hyperspy.api as hs
from ctypes import c_double
import csv
import concurrent.futures
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import font
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
print("Modules loaded.")

file = None
distances = None
currFunc = None

def setCurrFunc(funcName):
    global currFunc, file, distances
    currFunc = str(funcName)
    entry.delete(0, tk.END)
    if currFunc is "loadFile":
        entry.bind("<Return>", getEntry)
        label1['text'] = label1['text'] + "Please enter the path of the input file in the text box provided then press Enter.\n"
    elif currFunc is "toCSV":
        if file is None:
            label1['text'] = label1['text'] + "Please load a file before saving data.\n"
        elif distances is None:
            label1['text'] = label1['text'] + "Please analyze the file before saving data.\n"
        else:
            entry.bind("<Return>", getEntry)
            label1['text'] = label1['text'] + "Please enter the path of the file you want to save to in the text box provided then press Enter.\n"
    elif currFunc is "analysis":
        if file is None:
            label1['text'] = label1['text'] + "Please load a file before starting analysis.\n"
        else :
            entry.bind("<Return>", getEntry)
            label1['text'] = label1['text'] + "Please enter the number of rows and columns you would like to analyze, as integers, seperated by spaces. Press Enter when ready.\n"

def getEntry(event):
    global currFunc
    if currFunc is "loadFile":
        entry.unbind("<Return>")
        loadFile(entry.get())
    elif currFunc is "analysis":
        entry.unbind("<Return>")
        startAnalysis(entry.get())
    elif currFunc is "toCSV":
        entry.unbind("<Return>")
        toCSV(entry.get())
    
def loadFile(filename = None):
    global file
    label1['text'] = label1['text'] + "Loading file...\n"
    root.update()
    try:
        file = hs.load(filename)
        label1['text'] = label1['text'] + "File loaded.\n"
    except:
        label1['text'] = label1['text'] + "Error loading. Please check path and try again.\n"
    entry.delete(0, tk.END)
    #entry.unbind("<Return>")

def distance(x1, y1, x2, y2):
    return np.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

def intensity(values):
    s = ps.PixelatedSTEM(hs.signals.Signal2D(file.inav[values[0], values[1]]))
    imarray = np.array(s)
    distances[values[1]][values[0]] = imarray[values[3]][values[2]]

def findCenter(im, peak):
    center = (0,0)
    maximum = 0
    for (x,y) in np.ndenumerate(peak):
        for (a, b) in y:
            if (int(a) < len(im) and int(b) < len(im) and im[int(a)][int(b)] > maximum):
                maximum = im[int(a)][int(b)]
                center = (b, a)
    return center

def multiprocessing_func(values):
    global distances
    s = ps.PixelatedSTEM(hs.signals.Signal2D(file.inav[values[0], values[1]]))
    imarray = np.array(s)
    s = s.rotate_diffraction(0,show_progressbar=False)
    ############################################################################################################################
    st = s.template_match_disk(disk_r=5, lazy_result=False, show_progressbar=False)
    peak_array = st.find_peaks(lazy_result=False, show_progressbar=False)
    peak_array_com = s.peak_position_refinement_com(peak_array, lazy_result=False, show_progressbar=False)
    s_rem = s.subtract_diffraction_background(lazy_result=False, show_progressbar=False)
    peak_array_rem_com = s_rem.peak_position_refinement_com(peak_array_com, lazy_result=False, show_progressbar=False)
    ############################################################################################################################
    center = findCenter(imarray, peak_array_rem_com)

    # finds the specific spot and adding that distance to the array
    posDistance = 0
    closestPoint = center
        
    for (x,y) in np.ndenumerate(peak_array_rem_com):
        min = 999999
        for (a, b) in y:
            dis = distance(values[2], values[3], b, a)
            if dis < min:
                min = dis
                closestPoint = (b, a)
    posDistance = distance(closestPoint[0], closestPoint[1], center[0], center[1])
    distances[values[1]][values[0]] = round(posDistance, 2)

def startAnalysis(values = None):
    global file, currFunc
    pointxy = None
    methodOfAnalysis = ""
    if pointxy is None:
        def assignMethod(method):
            global methodOfAnalysis
            methodOfAnalysis = method
        def Mousecoords(event):
            global methodOfAnalysis
            pointxy = (int(event.x * 144 / 400), int(event.y * 144 / 400)) # get the mouse position from event
            l['text'] = l['text'] + str(pointxy[0]) + " " + str(pointxy[1]) + "\n"
            l['text'] = l['text'] + "Starting analysis...\n"
            r.update()
            analysis(pointxy, values, methodOfAnalysis)
            remove("temp.png")
            c2.unbind('<Button-1>')
            r.destroy()
            label1['text'] = label1['text'] + "Analysis complete.\n"

        s = ps.PixelatedSTEM(hs.signals.Signal2D(file.inav[0, 0]))
        s.save("temp.png")
        img = Image.open("temp.png")
        img = img.resize((400,400), Image.ANTIALIAS)
        img.save('temp.png')

        r = tk.Toplevel(root)

        c = tk.Canvas(r, height=720, width=1080)
        c.pack()
        f = tk.Frame(r, bg='#333333')
        f.place(relwidth=1, relheight=1)
        l = tk.Message(f, bg='#999999', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness = 0, bd=0, width = 1100)
        l.place(relx=0.05, rely=0.7, relwidth=0.9, relheight=0.2)
        b1 = tk.Button(f, text='Intensity Mapping', bg='#404040', font=('Calibri', 15), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: assignMethod("intensity"), pady=0.02, fg='#ffffff')
        b1.place(relx=0.2, rely=0.6, relwidth=0.2, relheight=0.05)
        b2 = tk.Button(f, text='Strain Mapping', bg='#404040', font=('Calibri', 15), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: assignMethod("strain"), pady=0.02, fg='#ffffff')
        b2.place(relx=0.6, rely=0.6, relwidth=0.2, relheight=0.05)
        c2 = tk.Canvas(r, width=400, height=400)
        c2.place(relx=0.3)
        img = ImageTk.PhotoImage(Image.open("temp.png"))
        c2.create_image(0, 0, anchor='nw', image=img)
        c2.bind('<Button-1>', Mousecoords)
        l['text'] = l['text'] + "Please click on the method of analysis and then the point you would like to analyze from the diffraction pattern above.\n"
        r.mainloop()
        
def analysis(pointxy, values, methodOfAnalysis):
    global file, distances
    t = values.split(" ")
    COL = int(t[1])
    ROW = int(t[0])

    list = []
    for r in range(ROW):
        for c in range(COL):
            list.append((r, c, pointxy[0], pointxy[1]))

    shared_array_base = multiprocessing.Array(c_double, ROW*COL)
    distances = np.ctypeslib.as_array(shared_array_base.get_obj())
    distances = distances.reshape(COL, ROW)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        if methodOfAnalysis is "strain":
            executor.map(multiprocessing_func, list)
        else:
            executor.map(intensity, list)
    entry.delete(0, tk.END)
    
def toCSV(filename = None):
    global distances
    f = open(filename, "w")
    writer = csv.writer(f)
    for i in distances:
        writer.writerow(i)
    f.close()
    label1['text'] = label1['text'] + "File saved.\n"
    entry.delete(0, tk.END)
    #entry.unbind("<Return>")

def barChart(INTERVAL = 0.01):

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    global distances
    if file is None:
        label1['text'] = label1['text'] + "Please load a file before creating a bar chart.\n"
    elif distances is None:
        label1['text'] = label1['text'] + "Please analyze the file before creating a bar chart.\n"
    else:
        label1['text'] = label1['text'] + "Creating bar chart. This might take several minutes depending on the size of data.\n"
        root.update()
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
        plt.xticks(y_pos, x_pos, fontsize = 5)
        plt.xlabel('Distance from center peek', fontsize = 5)
        plt.ylabel('Counts', fontsize = 5)
        plt.title('Distance Counts', fontsize = 5)
        plt.bar(y_pos, counts, align='center', alpha=0.95) # creates the bar plot

        axs = plt.gca()
        plt.setp(axs.get_xticklabels(), rotation=90)
        axs.tick_params(axis='x', which='major', labelsize=5)
        axs.tick_params(axis='y', which='major', labelsize=5)

        [l.set_visible(False) for (i,l) in enumerate(axs.xaxis.get_ticklabels()) if i % 2 != 0] 
        # The '2' is the every nth number of labels its shows on the x-axis. So rn is shows every 2nd label. 

        plt.gcf().subplots_adjust(bottom = 0.23)
        plt.rcParams["figure.dpi"] = 100
        #plt.savefig("temporary.png")
        
        barChartWindow = tk.Toplevel(root)
        barChartWindow.geometry('1280x720')
        figure = Figure(figsize=(11,11),dpi = 100)
        a = figure.add_subplot(111)
        a.bar(y_pos,counts)
        chart_type = FigureCanvasTkAgg(figure, barChartWindow)
        chart_type.draw()
        chart_type.get_tk_widget().place(relx=0.1, rely=0.1, relwidth=0.8, relheight=0.8)

        # r = tk.Tk()
        # c = tk.Canvas(r, height=720, width=1080)
        # c.pack()

        # img = tk.PhotoImage(master=c, file='temporary.png')
        # backLabel = tk.Label(r, image=img)
        # backLabel.place(relx=0, rely=0, relwidth=1, relheight=1)

        # r.mainloop()

def outlier(data):
    data = data.flatten()
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    min = q1 - (1.5 * iqr)
    max = q3 + (1.5 * iqr)
    return min, max

def heatMap():
    global distances
    if file is None:
        label1['text'] = label1['text'] + "Please load a file before creating a heat map.\n"
    elif distances is None:
        label1['text'] = label1['text'] + "Please analyze the file before creating a heat map.\n"
    else:
        data = distances.copy()
        df = pd.DataFrame(data, columns=np.arange(len(data[0])), index=np.arange(len(data)))
        data2 = distances.copy()
        min, max = outlier(data2)
        from matplotlib import cm as cm
        fig, a = plt.subplots(figsize=(6,5.5)) 
        yeet = sns.heatmap(df, cmap=cm.get_cmap("gray"),ax=a, vmin = min, vmax = max)
        fig = yeet.get_figure()

        heatMapWindow = tk.Toplevel(root)
        heatMapWindow.geometry('1280x720')
        chart_type = FigureCanvasTkAgg(fig, heatMapWindow)
        chart_type.draw()
        chart_type.get_tk_widget().pack()

if __name__ == "__main__":
    
    HEIGHT = 1080
    WIDTH = 1920

    root = tk.Tk()

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
    canvas.pack()
    frame = tk.Frame(root, bg='#333333')
    frame.place(relwidth=1, relheight=1)

    # Menu Label
    label = tk.Label(frame, text='Menu', bg='#333333', font=('Times New Roman', 50), fg='#ffffff')
    label.place(relx=0.40, rely=0.05, relwidth=0.2, relheight=0.05)

    # Text Output box
    label1 = tk.Message(frame, bg='#999999', font=('Calibri', 15), anchor='nw', justify='left', highlightthickness = 0, bd=0, width = 1500)
    label1.place(relx=0.1, rely=0.5, relwidth=0.8, relheight=0.35)

    # Entry box
    entry = tk.Entry(frame, font=('Calibri', 15))
    entry.place(relx=0.1, rely=0.9, relwidth=0.8, relheight=0.05)

    # Buttons
    button = tk.Button(frame, text='Load File', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: setCurrFunc("loadFile"), pady=0.02, fg='#ffffff')
    button.place(relx=0.42, rely=0.15, relwidth=0.16, relheight=0.05)

    button1 = tk.Button(frame, text='Start Analysis', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: setCurrFunc("analysis"), pady=0.02, fg='#ffffff')
    button1.place(relx=0.39, rely=0.22, relwidth=0.22, relheight=0.05)

    button2 = tk.Button(frame, text='Create Bar Chart', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: barChart(), pady=0.02, fg='#ffffff')
    button2.place(relx=0.375, rely=0.29, relwidth=0.25, relheight=0.05)

    button3 = tk.Button(frame, text='Create Heat Map', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: heatMap(), pady=0.02, fg='#ffffff')
    button3.place(relx=0.38, rely=0.36, relwidth=0.24, relheight=0.05)

    button4 = tk.Button(frame, text='Transfer Data to .csv', bg='#404040', font=('Calibri', 30), highlightthickness = 0, bd=0, activebackground='#666666', activeforeground='#ffffff', command=lambda: setCurrFunc("toCSV"), pady=0.02, fg='#ffffff')
    button4.place(relx=0.34, rely=0.43, relwidth=0.32, relheight=0.05)

    root.mainloop()
    if path.exists("temp.png"):
        remove("temp.png")