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
    area_thresholds = {'delamination': 2000, 'corrosion': 600} #if the largest damage area is smaller than this it will be considered hard

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,imgbin = cv2.threshold(imggray,labels[label]-1,255,cv2.THRESH_TOZERO)
    ret,imgbin = cv2.threshold(imgbin,labels[label],255,cv2.THRESH_TOZERO_INV)

    #pdb.set_trace()

    if np.unique(imgbin).shape == (1,): #this damage type is not present, therefore easy
    	return True

    #check opencv version
    (major, minor, _) = cv2.__version__.split(".")
    if major == 3:
        imgbin, contours, hierarchy = cv2.findContours(imgbin, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    else if major == 2:
    	contours, hierarchy = cv2.findContours(imgbin, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    else:
    	print("opencv version is fucked up")

    contour_areas = np.zeros(len(contours))

    at_leat_one_is_above_throshold = False

    for i,cont in enumerate(contours):
    	contour_areas[i] = cv2.contourArea(contours[i])
    	if(contour_areas[i]) > area_thresholds[label]:
    		at_leat_one_is_above_throshold = True



    return at_leat_one_is_above_throshold




def separate_dataset():

    root_path = '/home/teera/'
    mask_path = 'bridge_masks/'
    decks = ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']

    delamination_hard_file = open('{}{}delamination_hard.txt'.format(root_path, mask_path), 'w')
    delamination_easy_file = open('{}{}delamination_easy.txt'.format(root_path, mask_path), 'w')
    corrosion_hard_file = open('{}{}corrosion_hard.txt'.format(root_path, mask_path), 'w')
    corrosion_easy_file = open('{}{}corrosion_easy.txt'.format(root_path, mask_path), 'w')

    for deck in decks:
        os.chdir('{}{}{}'.format(root_path, mask_path, deck))
        for image in glob.glob('*.png'):
            img = cv2.imread('{}{}{}{}'.format(root_path, mask_path, deck, image))

            if is_easy(img, 'delamination') == False:
                print('{}{} is hard for delamination'.format(deck, image))
                delamination_hard_file.write('{}{}\n'.format(deck, image))
                
            else:
            	print('{}{} is easy for delamination'.format(deck, image))
                delamination_easy_file.write('{}{}\n'.format(deck, image))


            if is_easy(img, 'corrosion') == False:
                print('{}{} is hard for corrosion'.format(deck, image))
                corrosion_hard_file.write('{}{}\n'.format(deck, image))
                
            else:
            	print('{}{} is easy for corrosion'.format(deck, image))
                corrosion_easy_file.write('{}{}\n'.format(deck, image))



            
            
            


if __name__ == '__main__':
    separate_dataset()

