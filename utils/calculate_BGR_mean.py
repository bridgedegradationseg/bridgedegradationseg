import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import os.path as osp
import argparse

#import pdb

DATASET_BRIDGE_DIR = osp.expanduser('/root/fcn/bridgedegradationseg/dataset/')
#DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

#will return img as float32 (because that supports nan)
def mask_non_deck_fn(img, deck, treat_non_deck):

        if treat_non_deck == 'none':
            return img.astype('float32')

        assert deck.shape[0:2] == img.shape[0:2]
        assert img.shape[2] == 3
        assert len(deck.shape) == 2
        deck = deck/255  #so that deck is 1 or 0
        deck = np.repeat(deck[:,:,np.newaxis], 3, axis=2)  #duplicate deck into the 3rd dimension

        if treat_non_deck == 'black':
            img = img * deck.astype('uint8') 
            img = img.astype('float32')
        elif treat_non_deck == 'ignore':
            img = img.astype('float32')
            img[deck==0] = float('nan')     
        else:
            assert 1 == 0 #this should never happen

        return img


def calculate_BGR_mean(img_list, treat_non_deck):

    assert treat_non_deck in ['none', 'ignore', 'black']

    means = []
    imgcount = 0

    for file in open(img_list):
        file = file.strip()
        img_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_dataset/', '{}.jpg'.format(file))
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint32)
        if treat_non_deck != 'none':       
            deck_file = osp.join(DATASET_BRIDGE_DIR, 'deck_masks/', '{}.png'.format(file))
            deck = Image.open(deck_file)
            deck = np.array(deck, dtype=np.uint32)
        else: 
            deck = None

        img = mask_non_deck_fn(img, deck, treat_non_deck)

        img_mean = np.nanmean(img, axis=(0,1)) #mean in rgb
        if any(img_mean != img_mean): #check if mean is nan
            print('something is odd with image {}'.format(file))

        means.append(img_mean)
        imgcount += 1

    means = np.array(means)

    allmean = np.nanmean(means, axis=0)

    bgr_mean = allmean[::-1]

    print(bgr_mean)

    outfilepath = img_list + '_BGR_mean_{}.txt'.format(treat_non_deck)
    outfile = open(outfilepath, 'w')
    outfile.write(np.array2string(bgr_mean, separator=', '))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file with list of images to consider')
    parser.add_argument('-d', '--treat_non_deck', type=str, required=True, help='How to treat non-deck area ("none": no special treatment; "ignore": ignore non-deck; "black": black it out)')
    #label_list = '~/repos/bridgedegradationseg/dataset/bridge_masks/all_easy.txt'
    args = parser.parse_args()

    img_list = args.file
    calculate_BGR_mean(img_list, treat_non_deck=args.treat_non_deck)