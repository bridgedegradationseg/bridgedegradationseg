#!/bin/bash

cd /home/deeplearning/teera && \
rsync -ai --dry-run ubuntu@labelme-nii.mrteera.com:/home/ubuntu/bridge_dataset . > updated_filelist.txt && \
python /home/deeplearning/teera/bridgedegradationseg/utils/update_dataset.py && \
rsync -a ubuntu@labelme-nii.mrteera.com:/home/ubuntu/bridge_dataset . && \
rm -f /home/deeplearning/teera/bridge_masks/*/*.png && \
cat /home/deeplearning/teera/bridgedegradationseg/utils/get_labeled_images.m | matlab -nodisplay -nosplash -nodesktop -nojvm && \
python /home/deeplearning/teera/bridgedegradationseg/utils/separate_dataset.py
rm -f easy_bridge_masks.tar.bz2 && \
rm -f hard_bridge_masks.tar.bz2 && \
tar -cjSf easy_bridge_masks.tar.bz2 easy_bridge_masks && \
tar -cjSf hard_bridge_masks.tar.bz2 hard_bridge_masks && \
s3cmd put easy_bridge_masks.tar.bz2 s3://bridge-degradation/dataset/ && \
s3cmd put hard_bridge_masks.tar.bz2 s3://bridge-degradation/dataset/
