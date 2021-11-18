import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

NETWORK_ACCURACIES_FILEPATH = 'build/network_accuracies.csv'

ACCURACY_METRICS_NAME = 'mae' + '_score'

PLOT_SAVE_FILEPATH_TEMPLATE = 'build/train_graphics/{}_and_{}.png'

networks_data = pd.read_csv(NETWORK_ACCURACIES_FILEPATH)

def draw_accuracy_graph(param_x_name):
    global networks_data,ACCURACY_METRICS_NAME
    filtered_networks_data = networks_data[networks_data['experiment_sign'] == param_x_name]
    filtered_networks_data.plot.scatter(x=param_x_name, y=ACCURACY_METRICS_NAME)
    plt.savefig(PLOT_SAVE_FILEPATH_TEMPLATE.format(param_x_name, ACCURACY_METRICS_NAME))

for label, content in networks_data.items():
    if (not label in {ACCURACY_METRICS_NAME, 'Unnamed: 0','experiment_sign', 'loss_score'}):
        draw_accuracy_graph(label)
