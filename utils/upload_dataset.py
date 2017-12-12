# -*- coding: utf-8 -*-
    # TODO: remove open file 2 times

import boto3

s3 = boto3.resource('s3')

data = open('/home/deeplearning/teera/bridge_images.tar.bz2', 'rb')
s3.Bucket('bridge-degradation').put_object(Key='/dataset/bridge_images.tar.bz2', Body=data)
get_images_msg = "s3cmd get s3://bridge-degradation/dataset/bridge_images.tar.bz2"
