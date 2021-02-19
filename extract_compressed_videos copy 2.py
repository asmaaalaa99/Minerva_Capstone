"""
Extracts images from (compressed) videos, used for the FaceForensics++ dataset

Usage: see -h or https://github.com/ondyari/FaceForensics

Author: Andreas Roessler
Date: 25.01.2019
"""
import os
from os.path import join
import argparse
import subprocess
import cv2
from tqdm import tqdm
import os.path

from multiprocessing import Process


DATASET_PATHS = {
    'original': '/Volumes/MY PASSPORT/original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap'
}
COMPRESSION = ['c0', 'c23', 'c40']

data_path = "/Users/asmaaaly/Minerva_Shit/Capstone/Minerva_Capstone/data"

output_path = "/Users/asmaaaly/Minerva_Shit/Capstone/Minerva_Capstone/extracted"
def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)
    print("maybe it is working")
    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path,cv2.CAP_FFMPEG)
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                print("didn't work")
                break
            cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                        image)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))


def extract_method_videos(data_path, dataset, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, 'videos')
    images_path = join(data_path, DATASET_PATHS[dataset], compression, 'images')
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))

        
for dataset in DATASET_PATHS.keys():
    extract_method_videos(data_path,'original','c40')
 

