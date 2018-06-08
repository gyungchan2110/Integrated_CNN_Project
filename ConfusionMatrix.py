
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
import numpy as np
#print classification_report(np.argmax(y_pred,axis=1))

def plot_confusion_matrix(con_matrix, 
                          classes = '',
                          true_label='True label',
                          predict_label='Predict label',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          saveImgFile = False,
                          ImgFilename = None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    if normalize:
        con_matrix = con_matrix.astype('float') / con_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(con_matrix)

    plt.imshow(con_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = con_matrix.max() / 2.
    for i, j in itertools.product(range(con_matrix.shape[0]), range(con_matrix.shape[1])):
        plt.text(j, i, format(con_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if con_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel(true_label)
    plt.xlabel(predict_label) 
    if(saveImgFile):
        plt.savefig(ImgFilename, format='png')
    plt.show()




