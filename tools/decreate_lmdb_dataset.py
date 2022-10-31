import os
import sys
import re
import six
import math
import lmdb
import torch
import random
from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset

class lmdbChineseDataset(Dataset):

    def __init__(self, root=None, opt=None):

        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            self.filtered_index_list = [index + 1 for index in range(self.nSamples)]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index = index % self.nSamples
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = label.lower()

            if len(label) > 30:
                return self.__getitem__(np.random.randint(self.__len__()))

            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if True:
                    img = Image.open(buf).convert('RGB')
                else:
                    img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self.__getitem__(np.random.randint(self.__len__()))

        return (img, label, img_key)


if __name__ == '__main__':
    save_path = 'temp'
    lmdb_path = 'datasets_Chinese/scene_test'
    dataset = lmdbChineseDataset(lmdb_path)
    with open('tmp/Synth_train.txt','a') as f:
        for idx in range(dataset.__len__()):
            (img, lab, img_name) = dataset[idx]
            img_name = f"{img_name}.jpg"
            print(idx)
            f.write(f"{lab}\n")
            # img.save(os.path.join(save_path, img_name))
            