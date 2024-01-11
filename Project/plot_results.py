import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','no-latex'])
from sklearn.utils._testing import ignore_warnings
import warnings
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import os

your_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
#your_paath = "/Users/camillasancricca/Desktop/" # example of path


def mean(results_all, score):
    list_mean = []
    for res in results_all:
        list_mean.append(res[score])
    return list_mean


@ignore_warnings(category=DeprecationWarning)
def generateFigurePerformance(x_axis, xlabel, results_all, title, legend, encodings, score):

    plt.title(title)

    k = 0

    for i in range(0, len(results_all), 3):
        for j in range(0, len(encodings)):

            mean_perf = mean(results_all[i+j], score)

            plt.plot(x_axis, mean_perf, marker='o', label=legend[k] + " - " + encodings[j], markersize=2)
        k += 1

    plt.xlabel(xlabel)
    plt.ylabel(score)
    plt.legend(bbox_to_anchor=(0.9, -0.2))
    #plt.ylim(0.1, 2)  # if you want to fix a limit for the y_axis
    plt.savefig(your_path + title + ".pdf", bbox_inches='tight') # if you want to save the figure
    plt.show()


def plot(x_axis_values, x_label, results, title, algorithms, encodings, plot_type):

    title = str(title)

    if plot_type == "f1":
        generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, encodings, "f1")
    if plot_type == "precision":
        generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, encodings, "precision")
    if plot_type == "recall":
        generateFigurePerformance(x_axis_values, x_label, results, title, algorithms, encodings, "recall")
