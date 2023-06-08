# Installing modules from Requirements

import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable,'-m', 'pip', 'install', '-r', package])

path_package = os.path.join(os.getcwd(),'requirements.txt')
install(path_package)

# Importing modules

import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Face_Recognition():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def plot_images(self, images, titles, h, w, n_row, n_col):
        plt.figure(figsize=(2.2*n_col, 2.2*n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.20)
        for i in range(n_row*n_col):
            plt.subplot(n_row, n_col, i+1)
            plt.imshow(images[i].reshape((h,w)), cmap=plt.cm.gray)
            plt.title(titles[i])
            plt.xticks(())
            plt.yticks(())

    def number_images(self):
        total_images = 0
        for images in glob.glob(self.dataset_path + '/**/*', recursive=True):
            if images[-3:] == 'pgm' or images[-3:] == 'jpg':
                total_images += 1
        return total_images

    def number_identities(self):
        identity = os.listdir(self.dataset_path)
        return len(identity)

    def data_images(self):
        total_names = list()
        total_data = list()
        for folder in glob.glob(self.dataset_path + '/*'):
            name = folder.split('\\')[-1]

            for image in glob.glob(folder + '/*'):
                data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                total_data.append(data)
                total_names.append(name)
        return total_names, total_data

    def data_scaling(self,x,y,scale_type):
        total_resize_data = list()
        _, total_data = data_images()
        for images in total_data:
            resize_data = cv2.resize(images, (x,y), interpolation=cv2.INTER_LINEAR)
            total_resize_data.append(resize_data)
        return total_resize_data

    def data_normalization(self):
        _, total_data = data_images()
        total_normalize_data = list()
        for images in total_data:
            total_normalize_data.append(images/255)
        return total_normalize_data


