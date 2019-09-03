import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from datetime import datetime

class SingleDataset:
    def __init__(self, image_list:list, image_size:tuple, dataset_name:str, autosave=True):
        self.image_list = list(image_list)
        self.image_size = tuple(image_size)
        self.dataset_name = dataset_name
        self.autosave = autosave

        self.avg_histogram = np.zeros(256, dtype=np.float)
        self.date_refreshed = self._get_date_format()
        self.save_root = './saves/{}'.format(self.dataset_name)

        if not os.path.exists(self.save_root):
            os.mkdir(self.save_root)

    def _get_date_format(self):
        """
        Set time information for logging

        :return: time information (string)
        """
        nowdate = datetime.now()
        return "date{:04d}{:02d}{:02d}{:02d}".format(nowdate.year, nowdate.month, nowdate.day, nowdate.hour)

    def refresh_avg_histogram(self, image_size=None):
        """
        Set average grayscale histogram of image dataset
        Use single cpu only
        :param image_size: target image size, default size is assigned when init this class

        :return: None
        """
        if image_size == None:
            image_size = self.image_size

        hists = np.zeros(256, dtype=np.float)
        for i in self.image_list:
            img = cv2.imread(i)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, image_size)
            hists += cv2.calcHist([img], [0], None, [256], [0, 256]).reshape(256)

        hists /= len(self.image_list)

        self.date_refreshed = self._get_date_format()
        self.avg_histogram = hists
        if self.autosave:
            self.save()

    def plot_histogram(self, figname=None):
        """
        Plotting histogram

        :param figname: save path of bar plot figure, default None
        :return: None
        """
        plt.figure(figsize=(12,8))
        plt.bar([i for i in range(256)], self.avg_histogram)
        plt.title("histogram : {}, date : {}".format(self.dataset_name, self.date_refreshed))

        if not figname==None:
            savedir = os.path.join(self.save_root, self.date_refreshed)
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
            savename = 'exp_' + self.date_refreshed

        savedir = os.path.join(self.save_root, self.date_refreshed)
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        save_dict = {
            'name': self.dataset_name,
            'histogram': self.avg_histogram,
            'updated_date': self.date_refreshed,
            'image_size': self.image_size,
            'image_list': self.image_list,
            'autosave': self.autosave
        }

        import pickle
        with open(os.path.join(savedir, savename) + '.pkl', 'wb') as file:
            pickle.dump(save_dict, file, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    nowdate = datetime.now()
    print("date{:04d}{:02d}{:02d}".format(nowdate.year, nowdate.month, nowdate.day))