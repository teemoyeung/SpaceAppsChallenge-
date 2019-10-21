import csv
import numpy as np
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#creating data files: #for this example. we will use Mooselake.
falsefluc_file = open('false_fluctuations.txt', "w+")
falsefluc_file.write("Location: -94.21	57.71 \n")
falsefluc_file.write("False Fluctuation Data: \n")

geo_stormfile = open('geomagnetic.txt', "w+")
geo_stormfile.write("Location: -94.21	57.71 \n")
geo_stormfile.write("Geomagnetic Data: \n")

sharp_fluc = open('sharp_fluctuations.txt', "w+")
sharp_fluc.write("Location: -94.21	57.71 \n")
sharp_fluc.write("Sharp Fluctuation Data: \n")

df = pd.read_csv('dataset1.CSV', delimiter = ',')
df.head()
check_storm_array = [0] * 745 #to diffrenciate geostorm and non geostorm fluctuations

#for now, these points are hardcoded
x = df['1'] #first row; this always stays the same.
y = df['2'] #second row
y2 = df['4'] #nearby comparison

#working with the data
xs = []
for i in range(1,745,1):
    xs.append(i)

xs = np.array(xs, dtype=np.float64)
ys = np.array(y, dtype=np.float64)

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
         ((mean(xs) * mean(xs)) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b

m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]

def false_measurements(y, y2, check_storm_array, falsefluc_file): #take the average difference between two comparions
    sum_diff = 0
    for i in range (0, len(y)):
        sum_diff = sum_diff + abs(y[i] - y2[i])
    average_diff = sum_diff / len(y)
    print("average difference: ", average_diff)

    for i in range(0, len(y)):
        if ((abs(y[i] - y2[i])) > average_diff and (check_storm_array[i] != 1)):
            plt.axvspan(x[i], x[i + 1], facecolor='green', alpha=0.2)  # too much of a difference
            falsefluc_file.write("Time: %s " %x[i])
            falsefluc_file.write("Magnetic Fluctuation: %s\n" %y[i])
    falsefluc_file.close()

#to check the slope, sudden storm commencement (SSC) mush be above 10nT within 3 minute period.
def slope_check(x, y, check_storm_array):
    sum = 0
    average = 0
    difference = 0
    for i in range(0,len(x) - 1):
        if (((y[i + 1] - y[i]) / 20) > 10): #every interval is per hour, divididng it by 20 means per 3 minutes.
            plt.axvspan(x[i], x[i + 1], facecolor='orange', alpha=0.7)
            sharp_fluc.write(("Time: %s " % x[i]))
            sharp_fluc.write("Magnetic Fluctuation: %s\n" % y[i])
            if (((len(x) - i) > 25)): #for now, the code ignores the first 24 hours.
                for j in range(i, i + 21, 3):
                    difference = y[j:j+3].max() - y[j:j+3].min()
                    #print("diff", difference)
                    sum = sum + difference
                average = sum / 8
                sum = 0
                #print("average", average)
                if (average > 100):
                    plt.axvspan(x[i], x[i + 24], facecolor='red', alpha=0.2)  # KP is above 5 for a 24 hour period
                    for k in range (i, i+24):
                        check_storm_array[k] = 1
                        geo_stormfile.write(("Time: %s " %x[k]))
                        geo_stormfile.write("Magnetic Fluctuation: %s\n" %y[k])
    geo_stormfile.close()
    sharp_fluc.close()


#y[i + 24] - y[i]
#plotting code:
fig = plt.figure()
ax = fig.add_subplot(2,1,1,)
ax.set_yscale('log')

y_av = movingaverage(y, 2)
ax.plot(x, y_av, "b", label = 'Moose Lake, MB', linewidth = 0.5)

y_av2 = movingaverage(y2, 2)
ax.plot(x, y_av2, "g", label = 'Churchill, MB', linewidth = 0.5)

#False Measurement Checks:
slope_check(x, y, check_storm_array)
false_measurements(y, y2, check_storm_array, falsefluc_file)

style.use('ggplot')
ax.scatter(xs,ys,color='#003F72', label = 'Magnetic Intensity', marker = '*', s = 1)

#plt.plot(xs, regression_line)
plt.title('Magnetic Intensity')
xlabel('Data point over time')
ylabel('Intensity (log scale)')
leg = plt.legend( loc = 'upper right')
plt.show()
grid(True)