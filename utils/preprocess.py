import numpy as np
import os, sys
from PIL import Image
import progressbar

def save_patches(datapath, patch_size=100):

    banned_dirs = ['images_patches', 'bridge_masks_patches']
    print('Creating patches...')
    banned = False
    bar = progressbar.ProgressBar()
    for root, dirs, files in bar(os.walk(datapath)):
        for f in files:
            banned = False
            image_set = root.split('/')[-2]
            # Do not get into already extracted patches
            for element in banned_dirs:
                if element in root.split('/'):
                    banned = True
            # Ignoring weird file ending ._ (needs to get fixed, this is only quick hack)
            if f.endswith('.jpg') and not banned and not f.startswith('._'):
                # Sorry for next crappy line :)
                patches_path = os.path.join('/'.join(root.split('/')[:2]), image_set+'_patches', str(patch_size), root.split('/')[-1], f.split('.')[0])
                print(patches_path)
                if not os.path.exists(patches_path):
                    os.makedirs(patches_path)
                patches = extract_patches(os.path.join(root,f), patches_path, patch_size)



def extract_patches(imgpath, outpath, patch_size):
    im = Image.open(imgpath)
    for r in range(im.size[1] // patch_size):
        for c in range(im.size[0] // patch_size):
            subpatch = im.crop((c*patch_size, r*patch_size, (c+1)*patch_size, (r+1)*patch_size))
            subpatch.save(os.path.join(outpath,str(r)+'_'+str(c)+'.jpg'), quality=95)


def substract_channel_mean(X):
    # This is for a train shape of (numsamples, rows, cols, channels)
    return X - X.mean(axis=(0,-3,-2), keepdims=True)

def main():
    datapath = sys.argv[1]
    patch_size = int(sys.argv[2])
    save_patches(datapath, patch_size)

if __name__=="__main__":
    main()
