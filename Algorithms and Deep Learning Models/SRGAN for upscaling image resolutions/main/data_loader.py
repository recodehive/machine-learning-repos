import scipy.misc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    def __init__(self, dataset_name, img_res=(200, 200)):
        self.dataset_name = dataset_name
        self.img_res = img_res
    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"

        path = glob('../../datasets/%s/*' % (self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / 2), int(w / 2)

            img_hr = scipy.misc.imresize(img, self.img_res)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_hr, imgs_lr
    def load_pred(self, path):
        img = self.imread(path)
        imgs_hr = []
        imgs_lr = []
        h, w = self.img_res
        low_h, low_w = int(h / 2), int(w / 2)
        img_hr = scipy.misc.imresize(img, (self.img_res))
        img_lr = scipy.misc.imresize(img, (low_h, low_w))
        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        return imgs_hr, imgs_lr

    def load_resize(self, path):
        img = self.imread(path)
        imgs_hr = []
        imgs_lr = []
        h, w = self.img_res
        low_h, low_w = int(h/2), int(w/2)
        img_hr = scipy.misc.imresize(img, (self.img_res))
        img_lr = scipy.misc.imresize(img, (img_lr))
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.
        imgs_hr = np.resize(imgs_hr, (-1, 400,400,3))

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
