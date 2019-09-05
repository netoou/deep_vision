import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from datetime import datetime
from collections import OrderedDict

class SingleDataset:
    def __init__(self, image_list:list, image_size:tuple, dataset_name:str, autosave=True):
        self.image_list = list(image_list)
        self.image_size = tuple(image_size)
        self.dataset_name = dataset_name
        self.autosave = autosave

        self._avg_histogram = np.zeros(256, dtype=np.float)
        self._histograms = OrderedDict()
        self._avg_entropy = 0.0

        self._date_refreshed = self._get_date_format()
        self._save_root = './saves/{}'.format(self.dataset_name)

        if not os.path.exists(self._save_root):
            os.mkdir(self._save_root)

    def _get_date_format(self):
        """
        Set time information for logging

        :return: time information (string)
        """
        nowdate = datetime.now()
        return "date{:04d}{:02d}{:02d}{:02d}".format(nowdate.year, nowdate.month, nowdate.day, nowdate.hour)

    def set_avg_histogram(self, image_size=None, transform=None):
        """
        Set average grayscale histogram of image dataset
        Use single cpu only
        :param image_size: target image size, default size is assigned when init this class
        :param transform: image transform function to apply one image, not batch of images

        :return: None
        """
        if image_size == None:
            image_size = self.image_size

        hists = np.zeros(256, dtype=np.float)
        hist_dict = OrderedDict()
        for i in self.image_list:
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, image_size)

            if not transform == None:
                img = transform(img)

            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(256)
            hists += hist
            hist_dict[i] = hist

        hists /= len(self.image_list)

        self._date_refreshed = self._get_date_format()
        self._avg_histogram = hists
        self._histograms = hist_dict
        if self.autosave:
            self.save()

    def set_entropy(self):
        # TODO Complete function to calculate entropy of this dataset, use precomputed histogram
        n_pixels = self.image_size[0] * self.image_size[1]

        entropy = 0.0

        for key in self._histograms.keys():
            # loop for entire histograms of image
            hist = self._histograms[key]
            # entropy = - summation{p * log p}
            for k in hist:
                p_k = k / n_pixels
                entropy -= p_k * np.log2(p_k)

        entropy /= len(self._histograms.keys())
        self._avg_entropy = entropy

    def plot_histogram(self, figname=None):
        """
        Plotting histogram

        :param figname: save path of bar plot figure, default None
        :return: None
        """
        plt.figure(figsize=(12,8))
        plt.bar([i for i in range(256)], self.avg_histogram)
        plt.title("histogram : {}, date : {}".format(self.dataset_name, self._date_refreshed))

        if not figname==None:
            savedir = os.path.join(self._save_root, self._date_refreshed)
            if not os.path.exists(savedir):
                os.mkdir(savedir)

            plt.savefig(os.path.join(savedir,figname))

    def save(self, savename=None):
        """
        Save experiment result
        Now record only histogram
        :param savename: save file name
        :return: None
        """
        if savename==None:
            savename = 'exp_' + self._date_refreshed

        savedir = os.path.join(self._save_root, self._date_refreshed)
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        save_dict = {
            'name': self.dataset_name,
            'histogram': self._avg_histogram,
            'updated_date': self._date_refreshed,
            'image_size': self.image_size,
            'image_list': self.image_list,
            'autosave': self.autosave
        }

        import pickle
        with open(os.path.join(savedir, savename) + '.pkl', 'wb') as file:
            pickle.dump(save_dict, file, pickle.HIGHEST_PROTOCOL)
    @property
    def avg_histogram(self):
        return self._avg_histogram

    @property
    def avg_entropy(self):
        return self._avg_entropy



if __name__ == '__main__':
    print(12345)