import numpy as np
import os
import torch
from dataset import pklDataset
import cv2
import pickle
from PIL import Image
import argparse
from tqdm import tqdm
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video features')
    parser.add_argument('--video_path', type=str, default='tapvid_davis/tapvid_davis.pkl', help='path to video folder')
    parser.add_argument('--output_path', type=str, default='raw_video', help='path to save features')
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    imagedata = pklDataset(args.video_path)
    imagedata.switch_to(args.idx)
    # import pdb; pdb.set_trace()
    
    out_folder = os.path.join(args.output_path, imagedata.curname())
    os.makedirs(out_folder, exist_ok=True)
    
    count = 0
    for frame in tqdm(range(len(imagedata))):
        origin = imagedata.data[imagedata.seqnames[imagedata.curidx]]['video'][count]
        image = Image.fromarray(origin.astype(np.uint8))
        image_name = os.path.join(out_folder, '%05d.png'%count)
        image.save(image_name, format='PNG')
        count += 1
    
    
    

