#!/bin/bash

mkdir facenet
mkdir lightcnn

python3 download_gdrive.py 0B5MzpY9kBtDVZ2RpVDYwWmxoSUk facenet/20170512-110547.zip
python3 download_gdrive.py 1Jn6aXtQ84WY-7J3Tpr2_j6sX0ch9yucS lightcnn/LightCNN_29Layers_V2_checkpoint.pth.tar
