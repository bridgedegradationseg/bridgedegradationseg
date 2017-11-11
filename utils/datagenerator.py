import numpy as np
from utils.preprocess import crop_center
from PIL import Image
import os, sys

def dataset_to_txt(datapath):

    """
    Needs refactoring. Pass the bridge_masks path
    """
    paths = []
    bridge_masks_path = 'bridge_masks'
    bridge_images_path = 'images'
    for root, dirs, files in os.walk(datapath):
        if root.split('/')[-1] in ['deck_a', 'deck_c', 'deck_d', 'deck_e']:
            print(root)
            for f in files:
                maskpath = os.path.join(root, f)
                imgpath = os.path.join(root.split('/')[0], bridge_images_path, root.split('/')[-1], f.split('.')[-2]+'.jpg')
                paths.append(imgpath + "," + maskpath)
                
    with open(datapath+'dataset.txt', 'w') as txtfile:
        txtfile.write('\n'.join(paths))


def generator_from_txt(txtdatapath):
    while 1:
        f = open(txtdatapath)
        for line in f:
            imgpath, maskpath = line.rstrip().split(',')
            mask = Image.open(maskpath)
            mask = crop_center(mask)
            img = Image.open(imgpath)
            img = crop_center(img)

            yield np.expand_dims(np.array(img), axis=0), np.expand_dims(np.array(mask)[np.newaxis], axis=-1)


if __name__ == "__main__":
    
    datapath = sys.argv[1]
    # dataset_to_txt(datapath)
    a = generator_from_txt(datapath)
    for img, mask in a:
        print(img.shape,mask.shape)
        break
