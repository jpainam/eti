from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
from utils import read_json, write_json
import sys



class ETI(object):
    def __init__(self, root, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = root
        #self.train_dir = osp.join(self.dataset_dir, 'train')
        self.train_dir = root
        self.min_seq_len = min_seq_len
        self.split_train_json_path = osp.join(".", 'split_train.json')
        train, num_train_tracklets, num_train_scores, num_imgs_train = \
            self._process_dir(self.train_dir, self.split_train_json_path, relabel=True)

        num_imgs_per_tracklet = num_imgs_train
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_scores = num_train_scores
        num_total_tracklets = num_train_tracklets

        if verbose:
            print("=> ETI dataset loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset         | # ids | # tracklets")
            print("  ------------------------------")
            print("  train          | {:5d} | {:8d}".format(num_train_scores, num_train_tracklets))
            print("  ------------------------------")
            print("  total          | {:5d} | {:8d}".format(num_total_scores, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.num_train_scores = num_train_scores

    def _process_dir(self, dir_path, json_path, relabel):
        #if osp.exists(json_path):
        #    print("=> {} generated before, awesome!".format(json_path))
        #    split = read_json(json_path)
        #    return split['tracklets'], split['num_tracklets'], split['num_scores'], split['num_imgs_per_tracklet']
        pdirs = glob.glob(osp.join(dir_path, '*'))
        print("Processing {} with {} patient identities".format(dir_path, len(pdirs)))
        score_container = {}
        with open("data/global_scores", "r") as f:
            for line in f:
                score_container[line.split()[0]] = float(line.split()[1])
        #factor = 1.0 / sum(score_container.values())
        factor = 1.0 / 100.
        score_container = {k: v * factor for k, v in score_container.items()}
        tracklets = []
        num_imgs_per_tracklet = []
        img_paths = []
        for pdir in pdirs:
            score = score_container[os.path.basename(pdir)]
            tdirs = glob.glob(osp.join(pdir, '*'))
            for tdir in tdirs:
                raw_img_paths = glob.glob(osp.join(tdir, '*.jpg'))
                num_imgs = len(raw_img_paths)
                if num_imgs < self.min_seq_len:
                    continue
                num_imgs_per_tracklet.append(num_imgs)
                img_paths = []
                for img in raw_img_paths:
                    img_paths.append(img)
                    tracklets.append((img_paths, score))
        num_scores = len(score_container)
        num_tracklets = len(tracklets)

        print("Saving split to {}".format(json_path))
        split_dict = {
            'tracklets': tracklets,
            'num_tracklets': num_tracklets,
            'num_scores': num_scores,
            'num_imgs_per_tracklet': num_imgs_per_tracklet,
        }
        write_json(split_dict, json_path)

        return tracklets, num_tracklets, num_scores, num_imgs_per_tracklet


if __name__ == '__main__':
    # test
    dataset = ETI(root="/home/paul/eti/dataset")