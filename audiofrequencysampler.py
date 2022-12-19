from scipy.fft import rfft, rfftfreq
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import read
from PIL import Image
import simpleaudio as sa
import time
import keyboard
import colorsys
import turtle

#################################################################################################################
# WAV file frequency range sampler using fast forier transform with data filtering and smoothing adjustments
# for audio/music synced visual effects. 
#################################################################################################################

#Audio file and image output directories
soundfilepath = 'C:\\Users\\matth\\OneDrive\\Desktop\\macro2.wav' #Sound file used to calculate values
previewsoundfilepath = 'C:\\Users\\matth\\OneDrive\\Desktop\\macro.wav' #Sound file used for preview playback
outputfilepath = 'C:\\Users\\matth\\OneDrive\\Documents\\Frames\\night\\' #Image output directory
#Audio data
wave_object = sa.WaveObject.from_wave_file(previewsoundfilepath)
SAMPLE_RATE, stereodata = read(soundfilepath)
rawdata = np.array((stereodata[:,0] + stereodata[:,1]) / 2)
DURATION = rawdata.shape[0] / SAMPLE_RATE
maxfreq = int(SAMPLE_RATE / 2)
data = np.int16((rawdata / rawdata.max()) * 32767)
print('Max frequency ' + str(maxfreq))

#Adjustments
tolerance = 1000000 #Magnitude values beyond this threshold are calculated
samples = 1000 #Samples for all data
frequencyscalerange = [10, 25] #The entire frequency range for output
frequencyranges = [] #Specifies range(s) within frequencyscalerange used for output
ymax = 10000000 #Sets maximum value for frequency magnitude
ylimit = [0, 1] #Plot y-limit
sumgrouping = 0 #Combines frequencies and calculates sum of magnitudes. Can sometimes help remove noise
averagesums = False #When using sumgrouping, determines whether sums are averaged
cursorx = 1 #Axis for cursor when plotting
cursorincrement = 5 #Cursor increment
#Filter adjustments
usefilters = False
gaussian = 0 #Gaussian smoothing ammount
triangle = 0 #Triangle smoothing ammount
filterorder = [1] #Order filters are applied. 1:Gaussian, 2:Triangle 
#Color scale adjustments
usefreqstrengthasvalue = False
baseline = 0.1 #Baseline color value (must be > 0)
huedefault = 0 #Default color value at frequency 0
huemultiple = 40 #Color range multiple for wider range of colors
valuebaseline = 0.1 #Baseline value for brighter results
colorsmoothing = 3 #Smooth color values
setmax = 1 #Maximum color value
#Output
outputtype = 3 #1:Graph output, 2:Color scale preview window, 3:Color scale image output
# FOR GRAPH OUTPUT: Press space to pause playback, use left and right keys to control cursor,
# press i to mark index at cursor position, and press q to quit. Marked indexes are printed after quitting
#################################################################################################################

#Smoothing filters
def GaussianFilter(list, degree=3):
    if degree > 0:
        window = degree*2-1
        weight = np.array([1.0]*window)
        weightGauss = []
        for i in range(window):
            i = i-degree+1
            frac = i/float(window)
            gauss = 1/(np.exp((4*(frac))**2))
            weightGauss.append(gauss)
        weight = np.array(weightGauss)*weight
        smoothed = [0.0]*(len(list)-window)
        for i in range(len(smoothed)):
            smoothed[i] = sum(np.array(list[i:i+window])*weight)/sum(weight)
        return smoothed
    else:
        return list
def SmoothTriangle(data, degree):
    if degree > 0:
        triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
        smoothed=[]

        for i in range(degree, len(data) - degree * 2):
            point=data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed
if not usefilters:
    filterorder = []
if len(frequencyranges) == 0:
    frequencyranges = [0, maxfreq]

#Order frequency ranges from least to greatest
for i in range(0, len(frequencyranges), 2):
    start = frequencyranges[i]
    end = frequencyranges[i+1]
    if(start > end):
        frequencyranges[i+1] = start
        frequencyranges[i] = end

#Calculate data per sample and fft frequency range
samplewindow = int(len(data) / samples)
windowdata = []
windowduration = DURATION / samples
N = int(SAMPLE_RATE * windowduration)
xf = np.array(rfftfreq(N, 1 / SAMPLE_RATE)) #fft frequencies for audio file
normalizedxf = (xf / xf[-1])
frequencies = []
maxmag = 0

#Get the indexes of the closest frequencies specified by frequencyranges
frequencyindexranges = []
for i in range(0, len(frequencyranges)):
    freq = frequencyranges[i]
    frequencyindexranges.append(min(range(len(xf)), key=lambda i: abs(xf[i]-freq)))

#Calculate fft values for frequency magnitude for data within the index ranges and higher than set tolerance
percent = 0
start = time.time()
for i in range(0, samples):
    if percent < int((((i + 1) / samples)) * 100):
        print('Calculating fft samples ' + str(percent + 1) + '%')
    percent = int((((i + 1) / samples)) * 100)
    windowdata = data[int(i * samplewindow):int((i+1) * samplewindow)]
    allyf = np.array(np.abs(rfft(windowdata))) - tolerance #Full frequency magnitude range
    yf = [0] * len(allyf)
    #Get frequency magnitudes within specified indexes and remove negative numbers
    for x in range(0, len(allyf)):
        for y in range(0, len(frequencyindexranges), 2):
            startindex = frequencyindexranges[y]
            endindex = frequencyindexranges[y+1]
            if (x < endindex and x > startindex) and allyf[x] > 0:
                    yf[x] = np.clip(allyf[x], 0, ymax)
    #Get max value for window and max data for all samples
    windowmax = np.max(yf)
    if windowmax > maxmag:
        maxmag = windowmax
    #Group sums if sumgrouping > 0
    yfsums = yf
    if sumgrouping > 0:
        yfsums = []
        for x in range(0, len(yf), sumgrouping):
            if averagesums:
                yfsum = sum(yf[x:(x+sumgrouping)]) / sumgrouping
            else:
                yfsum = sum(yf[x:(x+sumgrouping)])
            for y in range(0, sumgrouping):
                yfsums.append(yfsum)
    frequencies.append(yfsums)
print('100%\nProcessed in ' + str(int(time.time() - start)) + 's')

#Apply filters in order specified
percent = 0
start = time.time()
for i in range(0, len(filterorder)):
    filter = filterorder[i]
    for x in range(0, len(frequencies)):
        if percent < int((x / len(frequencies)) * 100):
            print('Applying filter ' + str(i + 1) + '/' + str(len(filterorder)) + ' ' + str(percent + 1) + '%')
        percent = int((x / len(frequencies)) * 100)
        if filter == 1:
            frequencies[x] = GaussianFilter(frequencies[x], gaussian)
        elif filter == 2:
            frequencies[x] = SmoothTriangle(frequencies[x], triangle)
    print('Applying filter ' + str(i + 1) + '/' + str(len(filterorder)) + ' 100%')

#Normalize fft frequencies and values at 0-1 scale
frequencies = np.array(frequencies)
frequencies = frequencies / np.max(frequencies)
xf = xf[frequencyscalerange[0]:frequencyscalerange[1]]
normalizedxf = xf / np.max(xf)

#Generate output
if outputtype == 1:
    keypressdelay = 0.1
    playbackspeed = 1
    markedindexes = []
    closed = False
    while not closed:
        samlespersecond = (samples / DURATION) * playbackspeed
        play_object = wave_object.play()
        keypresstime = time.time()
        pausetime = time.time()
        start = time.time()
        paused = False
        sample = 0
        while sample < samples - 2 and not closed:
            if not paused:
                print('Sample: ' + str(sample) + '/' + str(samples))
                duration = time.time() - start
                sample = int(duration * samlespersecond)
            elif(keyboard.is_pressed('space') and (time.time() - keypresstime) > keypressdelay):
                keypresstime = time.time()
                start = time.time() - pauseduration
                paused = False
            plt.cla()
            plt.ylim(ylimit)
            plt.xlim(np.min(normalizedxf), np.max(normalizedxf))
            plt.plot(normalizedxf[:len(frequencies[sample])], frequencies[sample][frequencyscalerange[0]:frequencyscalerange[1]])
            cursorgraph = [-1] * len(normalizedxf)
            cursorgraph[cursorx] = 1
            for i in range(0, len(markedindexes)):
                cursorgraph[markedindexes[i]] = 1
            plt.plot(normalizedxf, cursorgraph)
            if keyboard.is_pressed('right') and (cursorx + cursorincrement) < frequencyscalerange[1]:
                cursorx += cursorincrement
            elif keyboard.is_pressed('left') and (cursorx - cursorincrement) > frequencyscalerange[0]:
                cursorx -= cursorincrement
            if (time.time() - keypresstime) > keypressdelay:
                if keyboard.is_pressed('space') and not paused:
                    keypresstime = time.time()
                    pauseduration = duration
                    paused = True
                elif keyboard.is_pressed('i'):
                    keypresstime = time.time()
                    markedindexes.append(cursorx)
            if keyboard.is_pressed('q'):
                closed = True
                break
            plt.pause(0.00000001)
        sa.stop_all()
    plt.close()
    print('Marked indexes: ' + str(markedindexes))
elif outputtype == 2 or 3:
    #Color scale output
    coloraverage = []
    frequenciesinrange = []
    for i in range(0, len(frequencies)):
        frequenciesinrange.append(frequencies[i][frequencyscalerange[0]:frequencyscalerange[1]])
    colorpersample = []
    for i in range(0, len(frequenciesinrange)):
        sample = frequenciesinrange[i]
        datasamples = len(sample)
        if usefreqstrengthasvalue:
            samplecolors = []
            r, g, b = [], [], []
            for x in range(0, len(sample)):
                hue = normalizedxf[x] * huemultiple
                value = sample[x]
                color = colorsys.hsv_to_rgb(hue + huedefault, 1, valuebaseline + (value * valuebaseline))
                r = (color[0] * value)
                g = (color[1] * value)
                b = (color[2] * value)
            samplecolors = [(baseline + r) / setmax, (baseline + g) / setmax, (baseline + b) / setmax]
            coloraverage.append(samplecolors)
        else:
            samplecolors = []
            hueweight = []
            r, g, b = [], [], []
            for x in range(0, len(sample)):
                value = baseline + sample[x]
                for y in range(0, int(value * 255)):
                    hueweight.append(normalizedxf[x] * huemultiple)
                hue = (sum(hueweight) / len(hueweight))
                color = colorsys.hsv_to_rgb(hue + huedefault, 1, 1)
                r = color[0]
                g = color[1]
                b = color[2]
            samplecolors = [r, g, b]
        coloraverage.append(samplecolors)
    if colorsmoothing > 0:
        r, g, b = [], [], []
        for x in range(0, len(coloraverage)):
            color = coloraverage[x]
            r.append(color[0])
            g.append(color[1])
            b.append(color[2])
        rsmoothed = GaussianFilter(r, colorsmoothing)
        gsmoothed = GaussianFilter(g, colorsmoothing)
        bsmoothed = GaussianFilter(b, colorsmoothing)
        smoothedcoloraverage = []
        for x in range(0, len(rsmoothed)):
            smoothedcoloraverage.append((rsmoothed[x], gsmoothed[x], bsmoothed[x]))
    else:
        smoothedcoloraverage = coloraverage
    if outputtype == 2:
        closed = False
        while not closed:
            wn = turtle.Screen()
            root = wn.getcanvas().winfo_toplevel()
            wn.title("Preview")
            wn.setup(width=500, height=500)
            wn.tracer(0)
            samlespersecond = (samples / DURATION)
            start = time.time()
            paused = False
            sample = 0
            play_object = wave_object.play()
            #Window close handler
            closed = False
            def on_close():
                global running
                closed = True
                wn.bye()
            root.protocol("WM_DELETE_WINDOW", on_close)
            while sample < samples and not closed:
                print('Sample: ' + str(sample) + '/' + str(samples))
                duration = time.time() - start
                sample = int(duration * samlespersecond)
                samplecolor = smoothedcoloraverage[sample]
                wn.bgcolor(samplecolor)
                wn.update()
                if closed:
                    break
    else:
        for i in range(0, len(smoothedcoloraverage)):
            color = smoothedcoloraverage[i]
            rgbcolor = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            filename = (str(i) + '.png')
            img = Image.new("RGB", (50,50), rgbcolor)
            img.save(outputfilepath + filename)
if outputtype == 4:
    frequenciesinrange = []
    valuesums = []
    for i in range(0, len(frequencies)):
        frequenciesinrange.append(frequencies[i][frequencyscalerange[0]:frequencyscalerange[1]])
    for i in range(0, len(frequenciesinrange)):
        valuesums.append(np.sum(frequenciesinrange[i]))
    min = np.min(valuesums)
    max = np.max(valuesums)
    np.array(valuesums)
    values = (valuesums / max) - min
    closed = False
    while not closed:
        wn = turtle.Screen()
        root = wn.getcanvas().winfo_toplevel()
        wn.title("Preview")
        wn.setup(width=500, height=500)
        wn.tracer(0)
        samlespersecond = (len(valuesums) / DURATION)
        start = time.time()
        paused = False
        sample = 0
        play_object = wave_object.play()
        #Window close handler
        closed = False
        def on_close():
            global running
            closed = True
            wn.bye()
        root.protocol("WM_DELETE_WINDOW", on_close)
        while sample < samples and not closed:
            print('Sample: ' + str(sample) + '/' + str(samples))
            duration = time.time() - start
            sample = int(duration * samlespersecond)
            value = values[sample]
            wn.bgcolor(value, value, value)
            wn.update()
            if closed:
                break
