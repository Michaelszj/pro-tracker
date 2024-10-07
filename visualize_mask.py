import torch
import glob
import os
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import cv2
def resize_tensor(tensor, size):
    # input tensor: (N, H, W, C)
    return F.interpolate(tensor.permute(0, 3, 1, 2)/255., size=size, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)*255.

path = './davis_dino'
videos = sorted(int(os.path.basename(i)) for i in(glob.glob(os.path.join(path, '*'))))
benchmark = pickle.load(open('./tapvid_davis/tapvid_davis.pkl','rb'))
names = list(benchmark.keys())

for i in tqdm(range(30)):
    video = videos[i]
    print(video)
    mask = os.path.join(path, str(video), 'dino_embeddings','sam2_mask','all_mask.pt')
    mask = torch.load(mask) # N,T,1,H,W
    mask = mask.permute(0,1,3,4,2).float() # N,T,H,W,1
    imgs = benchmark[names[i]]['video']
    H,W = imgs.shape[1:3]
    new_size = (480,(480*W)//H)
    resized = resize_tensor(torch.from_numpy(imgs).float().cuda(),new_size).cpu() # T,H,W,C
    # visualized = (resized[None,...] + mask*255.)/2. # N,T,H,W,C
    # visualized = visualized.cpu().numpy().astype(np.uint8)
    save_dir = os.path.join(path, str(video), 'dino_embeddings','sam2_mask')
    
    
    for j in range(mask.shape[0]):
        visualized = (resized + mask[j]*255.)/2. # T,H,W,C
        mask_j = visualized.cpu().numpy().astype(np.uint8) # T,H,W,C
        output_file = os.path.join(save_dir, f'mask_{j}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10  # 设置每秒帧数
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (W, H))
        for k in range(mask_j.shape[0]):
            frame = cv2.cvtColor(mask_j[k], cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
        video_writer.release()
        print(f'Saved {output_file}')
        
    
    
    