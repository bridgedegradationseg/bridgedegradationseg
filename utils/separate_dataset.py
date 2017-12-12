# -*- coding: utf-8 -*-

import glob
import os
from PIL import Image


def separate_dataset():
    pixel_threshold = 1600

    root_path = '/home/deeplearning/teera/'
    mask_path = 'bridge_masks/'
    decks = ['deck_a/', 'deck_c/', 'deck_d/', 'deck_e/']

    for deck in decks:
        os.chdir('{}{}{}'.format(root_path, mask_path, deck))
        for image in glob.glob('*.png'):
            im = Image.open('{}{}{}{}'.format(
                    root_path,
                    mask_path,
                    deck,
                    image)
                )
            black, grey, red, yellow, blue = 0, 0, 0, 0, 0

            for pixel in im.getdata():
                if pixel == (0, 0, 0):
                    black += 1
                elif pixel == (128, 128, 128):
                    grey += 1
                elif pixel == (255, 255, 128):
                    yellow += 1
                elif pixel == (128, 128, 255):
                    blue += 1
                elif pixel == (255, 128, 128):
                    red += 1
            colors = [black, grey, red, yellow, blue]
            is_easy = all(
                pixels >= pixel_threshold or pixels == 0 for pixels in colors
            )
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


if __name__ == '__main__':
    separate_dataset()
