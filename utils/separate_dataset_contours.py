# -*- coding: utf-8 -*-

import glob
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

import pdb

#findcontours can only handle a binary image, so we have to check for each category separately
def is_easy(img, label):


    labels = {'delamination': 226, 'corrosion': 76}  #other values that exist int the "raw" masks 166, 241, 'deck': 128, 
    area_thresholds = {'delamination': 2000, 'corrosion': 1} #if the largest damage area is smaller than this it will be considered hard

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,imgbin = cv2.threshold(imggray,labels[label]-1,255,cv2.THRESH_TOZERO)
    ret,imgbin = cv2.threshold(imgbin,labels[label],255,cv2.THRESH_TOZERO_INV)

    #pdb.set_trace()

    if np.unique(imgbin).shape == (1,): #this damage type is not present, therefore easy
    	return True

    imgbin, contours, hierarchy = cv2.findContours(imgbin, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)

    contour_areas = np.zeros(len(contours))

    at_leat_one_is_above_throshold = False

    for i,cont in enumerate(contours):
    	contour_areas[i] = cv2.contourArea(contours[i])
    	if(contour_areas[i]) > area_thresholds[label]:
    		at_leat_one_is_above_throshold = True



    return at_leat_one_is_above_throshold




def separate_dataset():
    pixel_threshold = 1600

    root_path = '/home/bridgedegradation/repos/bridgedegradationseg/dataset/'
    mask_path = 'easy_bridge_masks/'
    decks = ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']

    for deck in decks:
        os.chdir('{}{}{}'.format(root_path, mask_path, deck))
        for image in glob.glob('*.png'):
            img = cv2.imread('{}{}{}{}'.format(root_path, mask_path, deck, image))

            im = Image.open('{}{}{}{}'.format(root_path, mask_path, deck, image))#open it again with pillow for easy saving

            if is_easy(img, 'delamination') == False:
                print('{}{} is hard because of delamination\n'.format(deck, image))
                save_path = '{}{}{}'.format( root_path,'new_hard_bridge_masks/', deck)
                im.save('{}{}'.format(save_path, image))
            else:
                save_path = '{}{}{}'.format( root_path,'new_easy_bridge_masks/', deck)
                im.save('{}{}'.format(save_path, image))


            #if is_easy(img, 'corrosion') == False:
                #print('{}{} is hard because of corrosion\n'.format(deck, image))
                   
            
            
            


if __name__ == '__main__':
    separate_dataset()


"""
if is_easy:
save_path = '{}{}{}'.format(
    root_path,
    'easy_bridge_masks/',
    deck
)
im.save('{}{}'.format(save_path, image))
else:
save_path = '{}{}{}'.format(
    root_path,
    'hard_bridge_masks/',
    deck
)
im.save('{}{}'.format(save_path, image))
"""