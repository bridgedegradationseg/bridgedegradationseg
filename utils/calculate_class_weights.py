import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os.path as osp
import argparse

#import pdb

DATASET_BRIDGE_DIR = osp.expanduser('/root/fcn/bridgedegradationseg/dataset/')
#DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

def mark_non_deck_fn(lbl, deck, non_deck_lbl=42):
        assert len(deck.shape) == 2
        assert lbl.shape == deck.shape
        assert non_deck_lbl >= 0  #np.bincount cant handle negative numbers
        deck = deck/255  #so that deck is 1 or 0
        lbl[deck==0] = non_deck_lbl  #np.bincount cant handle negative numbers
        return lbl

def color_class_label(image):
        # https://stackoverflow.com/a/33196320
        color_codes = {
            (0, 0, 0): 0,
            (255, 255, 0): 1,
            (255, 0, 0): 2
        }
        
        color_map = np.ndarray(shape=(256*256*256), dtype='int32')
        color_map[:] = -1
        for rgb, idx in color_codes.items():
            rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
            color_map[rgb] = idx
        
        image = image.dot(np.array([65536, 256, 1], dtype='int32'))
        return color_map[image]

def calculate_class_weights(label_list, ignore_non_deck=True, n_classes=3):

    bins = np.zeros(n_classes+1)

    for file in open(label_list):
        file = file.strip()
        lbl_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_masks/', '{}.png'.format(file))
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.uint32)
        lbl = color_class_label(lbl)

        if ignore_non_deck:
            deck_file = osp.join(DATASET_BRIDGE_DIR, 'deck_masks/', '{}.png'.format(file))
            deck = Image.open(deck_file)
            deck = np.array(deck, dtype=np.uint32)
            lbl = mark_non_deck_fn(lbl, deck, non_deck_lbl=n_classes) #assign new label to non_deck
        bins += np.bincount(lbl.flatten(), minlength=n_classes+1)

    if np.count_nonzero(bins) != len(bins):
    	print('Warning! Some classes have zero occurence. Maybe n_classes is wrong?')
    	bins = bins[bins.nonzero()]

    if ignore_non_deck:
    	bins = bins[:-1]  #remove the count for non_deck, as we don't need it

    binsnorm = bins/bins.sum()
    weights = 1.0/binsnorm/len(binsnorm)

    print(weights)

    outfilepath = label_list + '_weights.txt'
    outfile = open(outfilepath, 'w')
    outfile.write(np.array2string(weights, separator=', '))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file with list of label iamges to consider')
    #label_list = '~/repos/bridgedegradationseg/dataset/bridge_masks/all_easy.txt'
    args = parser.parse_args()

    label_list = args.file
    calculate_class_weights(label_list, ignore_non_deck=True)