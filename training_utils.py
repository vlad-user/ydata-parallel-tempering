import sys
import itertools

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def dataset_splits(n_outputs):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., None] / 255., x_test[..., None] / 255.
    train_indices = np.argwhere(y_train < n_outputs).squeeze()
    test_indices = np.argwhere(y_test < n_outputs).squeeze()
    x_train, y_train = x_train[train_indices], y_train[train_indices]

    x_test, y_test = x_test[test_indices], y_test[test_indices]
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train)
    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)

def iteritems(items, batch_size):
    start_idx = 0
    while start_idx < items.shape[0]:
        yield items[start_idx: start_idx + batch_size]
        start_idx += batch_size

def create_empty_logs_dict(n_replicas):
    logs = {'error_' + str(i): [] for i in range(n_replicas)}
    logs.update({'loss_' + str(i): [] for i in range(n_replicas)})
    logs.update({'temperature_' + str(i): [] for i in range(n_replicas)})
    logs['swap_success'] = 0
    logs['swap_attempts'] = 0
    return logs

def append_to_log_dict(from_dict, to_dict):
    for metric, values in from_dict.items():
        if metric in ['swap_attempts', 'swap_success']:
            to_dict[metric] += values
        elif ('error' in metric
              or 'loss' in metric
              or 'temperature' in metric):
            to_dict[metric].append(values)
        else:
            print(metric)
            raise ValueError(metric)
    return to_dict

def print_logs(logs, epoch, n_epochs, step):
    n_replicas = len([k for k in logs.keys() if 'loss_' in k])
    errors = [logs['error_' + str(i)] for i in range(n_replicas)]
    bufs = ['{0:.4f}' for i in range(n_replicas)]
    bufs = [b.format(e[-1]) for b, e in zip(bufs, errors)]
    bufs = ['{}-->'.format(i) + b for i, b in enumerate(bufs)]
    buf = '[' + '|'.join(bufs) + ']'
    buf = '[step: {}]'.format(step) + buf
    buf = '[epoch: {0}/{1}]'.format(epoch, n_epochs) + buf
    buf = buf + '[{0:.4f}]'.format(logs['swap_success'] / logs['swap_attempts'])
    sys.stdout.write('\r' + buf)
    sys.stdout.flush()

def plot_mixing(logs, noise_list, n_replicas, epochs):
    fig, ax = plt.subplots(figsize=(16, 8))

    noise_list = [float('{0:.5f}'.format(n)) for n in noise_list]

    key_map = {n:i for i, n in enumerate(noise_list)}
    key_map.update({i: n for i, n in enumerate(noise_list)})

    for r in range(n_replicas):
        y = itertools.chain(*logs['train']['temperature_' + str(r)])
        y = [float('{0:.5f}'.format(t)) for t in y]
        x = np.linspace(0, epochs, len(y))

        y_new = [key_map[i] for i in y]
        ax.plot(x, y_new, label='replica ' + str(r),
                  linewidth=6
               )
    yticks_names = [float("{0:.5f}".format(b)) for b in noise_list]

    plt.gca().set_yticklabels(['0'] + yticks_names)
    plt.xlabel('EPOCHS', fontsize=25)
    plt.ylabel('DROPOUT RATE', fontsize=25)
    plt.xticks(fontsize=25, rotation=0)
    plt.yticks(fontsize=25, rotation=0)
    leg = plt.legend(bbox_to_anchor=(1, 1.1), prop={'size': 20})
    leg.get_frame().set_linewidth(6.0)