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
            if f.endswith('.jpg') or f.endswith('.png') and not banned and not f.startswith('._'):
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
            subpatch.save(os.path.join(outpath,str(r)+'_'+str(c)+'.png'))


def substract_channel_mean(X):
    # This is for a train shape of (numsamples, rows, cols, channels)
    return X - X.mean(axis=(0,-3,-2), keepdims=True)

def main():
    datapath = sys.argv[1]
    patch_size = int(sys.argv[2])
    save_patches(datapath, patch_size)


def load_full_images(datapath):
	bridge_masks_path = 'bridge_masks'
	bridge_images_path = 'images'
	imgs = []
	masks = []
	for root, dirs, files in os.walk(datapath):
            if root.split('/')[-1] in ['deck_a', 'deck_c', 'deck_d', 'deck_e']:
                print(root)
                for f in files:
                        mask = Image.open(os.path.join(root, f))
                        mask = crop_center(mask)
                        imgpath = os.path.join(root.split('/')[0], bridge_images_path, root.split('/')[-1], f.split('.')[-2]+'.jpg')
                        img = Image.open(imgpath)
                        img = crop_center(img)
                        mask = np.asarray(mask)
                        mask.flags.writeable = True
                        masks.append(mask)
                        img = np.asarray(img)
                        img.flags.writeable = True
                        imgs.append(img)
	return imgs, masks


def crop_center(img):
    half_width = img.size[0] // 2
    half_height = img.size[1] // 2

    w = 1024//2
    h = 768//2

    cropped_img = img.crop(
            (
                half_width - w,
                half_height - h,
                half_width + w,
                half_height + h
            )
        )
    return cropped_img

if __name__=="__main__":
    main()
