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
    if cv2.__version__.split(".")[0] == '3':
        imgbin, contours, hierarchy = cv2.findContours(imgbin, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    elif cv2.__version__.split(".")[0] == '2':
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

    #root_path = '/home/deeplearning/teera/'
    root_path = '/home/bridgedegradation/repos/bridgedegradationseg/dataset/'
    mask_path = 'bridge_masks/'
    decks = ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']

    delamination_hard_file = open('{}{}delamination_hard.txt'.format(root_path, mask_path), 'w')
    delamination_easy_file = open('{}{}delamination_easy.txt'.format(root_path, mask_path), 'w')
    corrosion_hard_file = open('{}{}corrosion_hard.txt'.format(root_path, mask_path), 'w')
    corrosion_easy_file = open('{}{}corrosion_easy.txt'.format(root_path, mask_path), 'w')
    all_hard_file = open('{}{}all_hard.txt'.format(root_path, mask_path), 'w')
    all_easy_file = open('{}{}all_easy.txt'.format(root_path, mask_path), 'w')

    corrosion_easy_count = 0
    corrosion_hard_count = 0
    delamination_easy_count = 0
    delamination_hard_count = 0
    all_easy_count = 0
    all_hard_count = 0

    for deck in decks:
        os.chdir('{}{}{}'.format(root_path, mask_path, deck))
        for image in glob.glob('*.png'):
            img = cv2.imread('{}{}{}{}'.format(root_path, mask_path, deck, image))

            all_easy = True
            all_hard = True

            if is_easy(img, 'delamination') == False:
                #print('{}{} is hard for delamination'.format(deck, image))
                delamination_hard_count += 1
                delamination_hard_file.write('{}{}\n'.format(deck, image))
                all_easy = False
                
            else:
                #print('{}{} is easy for delamination'.format(deck, image))
                delamination_easy_count += 1
                delamination_easy_file.write('{}{}\n'.format(deck, image))
                all_hard = False


            if is_easy(img, 'corrosion') == False:
                #print('{}{} is hard for corrosion'.format(deck, image))
                corrosion_hard_count += 1
                corrosion_hard_file.write('{}{}\n'.format(deck, image))
                all_easy = False
                
            else:
                #print('{}{} is easy for corrosion'.format(deck, image))
                corrosion_easy_count += 1
                corrosion_easy_file.write('{}{}\n'.format(deck, image))
                all_hard = False


            if all_easy:
                all_easy_count += 1
                all_easy_file.write('{}{}\n'.format(deck, image))

            if all_hard:
                all_hard_count += 1
                all_hard_file.write('{}{}\n'.format(deck, image))


    print("Delamintaion: {} hard, {} easy. Corrosion: {} hard, {} easy.    {} are easy for both. {} are hard for both.\n".format(delamination_hard_count, delamination_easy_count, corrosion_hard_count, corrosion_easy_count, all_easy_count, all_hard_count))



            
            
            


if __name__ == '__main__':
    separate_dataset()

