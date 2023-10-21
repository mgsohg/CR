from __future__ import division, print_function
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import itertools
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from torchvision import models, transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
from imutils import paths
import os
import cv2
import matplotlib.pyplot as plt
import os
import copy
from sklearn.metrics import confusion_matrix


def make_confusion_matrix(cf, group_names=None,categories='auto',count=True,percent=True,cbar=False,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    plt.style.use(['science', 'ieee', 'no-latex'])
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.4f}\nPrecision={:0.4f}\nRecall={:0.4f}\nF1 Score={:0.4f}".format(
                accuracy, precision, recall, f1_score)
            print(stats_text)
        else:
            stats_text = "\n\nAccuracy={:0.4f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize == None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False
    csfont = {'fontname': 'Times New Roman'}
    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=True,xticklabels=categories, yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label',fontsize=10,style='italic',**csfont)
        #plt.xlabel('Predicted label' + stats_text)
        plt.xlabel('Predicted label',fontsize=10,style='italic',**csfont)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title,fontsize=12,**csfont)
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    from matplotlib.colors import ListedColormap
    #cmap = ListedColormap(['white'])

    # print(cm)
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    # plt.colorbar()
    # tick_marks = np.arange(len(classes))
    # plt.xticks(tick_marks, classes, rotation=45)
    # plt.yticks(tick_marks, classes)
    #
    # fmt = '.2f'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.tight_layout()
    # plt.show()

    cf_matrix = cm
    #sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True,fmt='.2%', cmap='Blues')
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['B', 'M']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    #sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    #plt.show()
    make_confusion_matrix(cm,
                          group_names=group_names,
                          categories=categories,
                          cmap = 'binary',title='BUSI fold1')



class URepNetv1(nn.Module):
    def __init__(self):
        super(URepNetv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = torch.nn.ReLU()
        self.conv1_1 = nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1)
        self.batch1_1 = nn.BatchNorm2d(16)
        self.relu1_1 = torch.nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = torch.nn.ReLU()
        self.conv2_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.batch2_1 = nn.BatchNorm2d(32)
        self.relu2_1 = torch.nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)  # 56

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(64)
        self.relu3 = torch.nn.ReLU()
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.batch3_1 = nn.BatchNorm2d(64)
        self.relu3_1 = torch.nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)  # 28

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(128)
        self.relu4 = torch.nn.ReLU()
        self.conv4_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.batch4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = torch.nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)  # 14

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(256)
        self.relu5 = torch.nn.ReLU()
        self.conv5_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.batch5_1 = nn.BatchNorm2d(256)
        self.relu5_1 = torch.nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)  # 7

        self.conv6 = nn.Conv2d(256, 512, kernel_size=7, stride=1)  # 1x1
        self.batch6 = nn.BatchNorm2d(512)
        self.relu6 = torch.nn.ReLU()

        # deconv

        self.Convt1 = nn.ConvTranspose2d(512, 256, kernel_size=7, stride=1)
        self.batch_trans = nn.BatchNorm2d(256)
        self.relu_trans = torch.nn.ReLU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(128)
        self.relu7 = torch.nn.ReLU()
        self.conv7_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batch7_1 = nn.BatchNorm2d(128)
        self.relu7_1 = torch.nn.ReLU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # 28

        self.conv8 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(64)
        self.relu8 = torch.nn.ReLU()
        self.conv8_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch8_1 = nn.BatchNorm2d(64)
        self.relu8_1 = torch.nn.ReLU()
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(32)
        self.relu9 = torch.nn.ReLU()
        self.conv9_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batch9_1 = nn.BatchNorm2d(32)
        self.relu9_1 = torch.nn.ReLU()
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.batch10 = nn.BatchNorm2d(16)
        self.relu10 = torch.nn.ReLU()
        self.conv10_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.batch10_1 = nn.BatchNorm2d(16)
        self.relu10_1 = torch.nn.ReLU()
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        #E1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.batch1_1(x)
        x = self.relu1_1(x)
        x, indices1 = self.pool1(x)

        # E2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2_1(x)
        x = self.batch2_1(x)
        x = self.relu2_1(x)
        x, indices2 = self.pool2(x)

        # E3
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x, indices3 = self.pool3(x)

        # E4
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.batch4_1(x)
        x = self.relu4_1(x)
        x, indices4 = self.pool4(x)

        # E5
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        x = self.conv5_1(x)
        x = self.batch5_1(x)
        x = self.relu5_1(x)
        x, indices5 = self.pool5(x)

        #bridge
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        latent = x

        # D1
        x = self.Convt1(x)
        x = self.batch_trans(x)
        x = self.relu_trans(x)
        x = self.unpool1(x, indices=indices5)

        # D2
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        x = self.conv7_1(x)
        x = self.batch7_1(x)
        x = self.relu7_1(x)
        x = self.unpool2(x, indices=indices4)

        # D3
        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)
        x = self.conv8_1(x)
        x = self.batch8_1(x)
        x = self.relu8_1(x)
        x = self.unpool3(x, indices=indices3)

        # D4
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.relu9(x)
        x = self.conv9_1(x)
        x = self.batch9_1(x)
        x = self.relu9_1(x)
        x = self.unpool4(x, indices=indices2)

        # D5
        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        x = self.conv10_1(x)
        x = self.batch10_1(x)
        x = self.relu10_1(x)
        x = self.unpool5(x, indices=indices1)  # 14x14

        x = self.conv11(x)

        return x, latent


class URepNetv2(nn.Module):
    def __init__(self):
        super(URepNetv2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1)
        self.batch1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = torch.nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.batch2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = torch.nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)  # 56

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.batch3_1 = nn.BatchNorm2d(128)
        self.relu3_1 = torch.nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)  # 28

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()
        self.conv4_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.batch4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = torch.nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)  # 14

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(512)
        self.relu5 = torch.nn.ReLU()
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.batch5_1 = nn.BatchNorm2d(512)
        self.relu5_1 = torch.nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)  # 7

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=7, stride=1)  # 1x1
        self.batch6 = nn.BatchNorm2d(1024)
        self.relu6 = torch.nn.ReLU()

        # deconv

        self.Convt1 = nn.ConvTranspose2d(1024, 512, kernel_size=7, stride=1)
        self.batch_trans = nn.BatchNorm2d(512)
        self.relu_trans = torch.nn.ReLU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(256)
        self.relu7 = torch.nn.ReLU()
        self.conv7_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch7_1 = nn.BatchNorm2d(256)
        self.relu7_1 = torch.nn.ReLU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)  # 28

        self.conv8 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(128)
        self.relu8 = torch.nn.ReLU()
        self.conv8_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batch8_1 = nn.BatchNorm2d(128)
        self.relu8_1 = torch.nn.ReLU()
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(64)
        self.relu9 = torch.nn.ReLU()
        self.conv9_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch9_1 = nn.BatchNorm2d(64)
        self.relu9_1 = torch.nn.ReLU()
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.batch10 = nn.BatchNorm2d(32)
        self.relu10 = torch.nn.ReLU()
        self.conv10_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.batch10_1 = nn.BatchNorm2d(32)
        self.relu10_1 = torch.nn.ReLU()
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        #E1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.batch1_1(x)
        x = self.relu1_1(x)
        x, indices1 = self.pool1(x)

        # E2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2_1(x)
        x = self.batch2_1(x)
        x = self.relu2_1(x)
        x, indices2 = self.pool2(x)

        # E3
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x, indices3 = self.pool3(x)

        # E4
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.batch4_1(x)
        x = self.relu4_1(x)
        x, indices4 = self.pool4(x)

        # E5
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        x = self.conv5_1(x)
        x = self.batch5_1(x)
        x = self.relu5_1(x)
        x, indices5 = self.pool5(x)

        #bridge
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        latent = x

        # D1
        x = self.Convt1(x)
        x = self.batch_trans(x)
        x = self.relu_trans(x)
        x = self.unpool1(x, indices=indices5)

        # D2
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        x = self.conv7_1(x)
        x = self.batch7_1(x)
        x = self.relu7_1(x)
        x = self.unpool2(x, indices=indices4)

        # D3
        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)
        x = self.conv8_1(x)
        x = self.batch8_1(x)
        x = self.relu8_1(x)
        x = self.unpool3(x, indices=indices3)

        # D4
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.relu9(x)
        x = self.conv9_1(x)
        x = self.batch9_1(x)
        x = self.relu9_1(x)
        x = self.unpool4(x, indices=indices2)

        # D5
        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        x = self.conv10_1(x)
        x = self.batch10_1(x)
        x = self.relu10_1(x)
        x = self.unpool5(x, indices=indices1)  # 14x14

        x = self.conv11(x)

        return x, latent


class URepNetv3(nn.Module):
    def __init__(self):
        super(URepNetv3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=1)
        self.batch1 = nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.conv1_1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
        self.batch1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = torch.nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.batch2 = nn.BatchNorm2d(128)
        self.relu2 = torch.nn.ReLU()
        self.conv2_1 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1)
        self.batch2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = torch.nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.batch3 = nn.BatchNorm2d(256)
        self.relu3 = torch.nn.ReLU()
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.batch3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = torch.nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.batch4 = nn.BatchNorm2d(512)
        self.relu4 = torch.nn.ReLU()
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.batch4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = torch.nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv5 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, stride=1)
        self.batch5 = nn.BatchNorm2d(1024)
        self.relu5 = torch.nn.ReLU()
        self.conv5_1 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=1)
        self.batch5_1 = nn.BatchNorm2d(1024)
        self.relu5_1 = torch.nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv6 = nn.Conv2d(1024, 2048, kernel_size=7, stride=1)  # 1x1
        self.batch6 = nn.BatchNorm2d(2048)
        self.relu6 = torch.nn.ReLU()

        # deconv

        self.Convt1 = nn.ConvTranspose2d(2048, 1024, kernel_size=7, stride=1)
        self.batch_trans = nn.BatchNorm2d(1024)
        self.relu_trans = torch.nn.ReLU()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.batch7 = nn.BatchNorm2d(512)
        self.relu7 = torch.nn.ReLU()
        self.conv7_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.batch7_1 = nn.BatchNorm2d(512)
        self.relu7_1 = torch.nn.ReLU()
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.batch8 = nn.BatchNorm2d(256)
        self.relu8 = torch.nn.ReLU()
        self.conv8_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.batch8_1 = nn.BatchNorm2d(256)
        self.relu8_1 = torch.nn.ReLU()
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.batch9 = nn.BatchNorm2d(128)
        self.relu9 = torch.nn.ReLU()
        self.conv9_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.batch9_1 = nn.BatchNorm2d(128)
        self.relu9_1 = torch.nn.ReLU()
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.batch10 = nn.BatchNorm2d(64)
        self.relu10 = torch.nn.ReLU()
        self.conv10_1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.batch10_1 = nn.BatchNorm2d(64)
        self.relu10_1 = torch.nn.ReLU()
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        #self.sig = nn.Sigmoid()

    def forward(self, x):
        #E1
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.relu1(x)
        x = self.conv1_1(x)
        x = self.batch1_1(x)
        x = self.relu1_1(x)
        x, indices1 = self.pool1(x)

        # E2
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        x = self.conv2_1(x)
        x = self.batch2_1(x)
        x = self.relu2_1(x)
        x, indices2 = self.pool2(x)

        # E3
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.relu3(x)
        x = self.conv3_1(x)
        x = self.batch3_1(x)
        x = self.relu3_1(x)
        x, indices3 = self.pool3(x)

        # E4
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.relu4(x)
        x = self.conv4_1(x)
        x = self.batch4_1(x)
        x = self.relu4_1(x)
        x, indices4 = self.pool4(x)

        # E5
        x = self.conv5(x)
        x = self.batch5(x)
        x = self.relu5(x)
        x = self.conv5_1(x)
        x = self.batch5_1(x)
        x = self.relu5_1(x)
        x, indices5 = self.pool5(x)

        #bridge
        x = self.conv6(x)
        x = self.batch6(x)
        x = self.relu6(x)
        latent = x

        # D1
        x = self.Convt1(x)
        x = self.batch_trans(x)
        x = self.relu_trans(x)
        x = self.unpool1(x, indices=indices5)

        # D2
        x = self.conv7(x)
        x = self.batch7(x)
        x = self.relu7(x)
        x = self.conv7_1(x)
        x = self.batch7_1(x)
        x = self.relu7_1(x)
        x = self.unpool2(x, indices=indices4)

        # D3
        x = self.conv8(x)
        x = self.batch8(x)
        x = self.relu8(x)
        x = self.conv8_1(x)
        x = self.batch8_1(x)
        x = self.relu8_1(x)
        x = self.unpool3(x, indices=indices3)

        # D4
        x = self.conv9(x)
        x = self.batch9(x)
        x = self.relu9(x)
        x = self.conv9_1(x)
        x = self.batch9_1(x)
        x = self.relu9_1(x)
        x = self.unpool4(x, indices=indices2)

        # D5
        x = self.conv10(x)
        x = self.batch10(x)
        x = self.relu10(x)
        x = self.conv10_1(x)
        x = self.batch10_1(x)
        x = self.relu10_1(x)
        x = self.unpool5(x, indices=indices1)  # 14x14

        x = self.conv11(x)

        return x, latent



