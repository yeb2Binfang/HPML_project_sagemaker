import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# data = pd.read_excel('/Users/panxy/Desktop/Parallel and Customized Architecture/Project/statistics.xlsx',sheet_name='all')

data = pd.read_csv('data.csv')

def plot_histogram(data):

    plt.style.use('seaborn-deep')
    w = 0.4
    bar = np.arange(len(data['net'].unique()))
    for index, machine in enumerate(data['machine'].unique()):
        cur = data[(data['machine'] == machine)][['net', 'time']]
        tmp = plt.bar([i + w * index for i in bar], cur['time'], w, label=machine)
        plt.bar_label(tmp)

    for net in data['net'].unique():
        print(net,float(data[(data['net']==net)&(data['machine']=='hpc')]['time'])\
              /float(data[(data['net']==net)&(data['machine']=='sagemaker')]['time']))

    plt.xticks(bar, data['net'].unique())
    plt.xlabel('Net')
    plt.ylabel('Time')
    plt.legend(loc='upper left')
    plt.show()

def plot_accuracy(data):

    plt.style.use('seaborn-deep')
    w = 0.4
    bar = np.arange(len(data['net'].unique()))
    colors = ['orange','purple']
    for index, machine in enumerate(data['machine'].unique()):
        cur = data[(data['machine'] == machine)][['net', 'accuracy']]
        tmp = plt.bar([i + w * index for i in bar], cur['accuracy'], w, label=machine,color=colors[index])
        plt.bar_label(tmp)

    plt.xticks(bar, data['net'].unique())
    plt.xlabel('Net')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper center')
    plt.show()



plot_histogram(data)
plot_accuracy(data)
