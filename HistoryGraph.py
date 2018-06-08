
# coding: utf-8

import matplotlib.pyplot as plt


def draw_history_graph(model_history, imgPath = None):
    fig, loss_ax = plt.subplots()

    acc_ax = loss_ax.twinx()

    loss_ax.plot(model_history.history['loss'], 'y', label='train loss')
    loss_ax.plot(model_history.history['val_loss'], 'r', label='val loss')

    acc_ax.plot(model_history.history['acc'], 'b', label='train acc')
    acc_ax.plot(model_history.history['val_acc'], 'g', label='val acc')

    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')

    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    if(imgPath):
        plt.savefig(imgPath)
    plt.show()
    