import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import pickle
path = './davis_dino'
videos = sorted(int(os.path.basename(i)) for i in(glob.glob(os.path.join(path, '*'))))
selected = [[0], [4], [1], [5], [10], [0], [0, 10], [5, 9], [0], [0], 
            [2, 16, 20], [0, 4], [0], [10, 18], [1], [1], [0], [1], [1], [10], 
            [1, 5], [15], [1, 5], [5], [0], [0], [0], [5], [5, 13], [0, 7, 12]]

for i in tqdm(range(30)):
    video = videos[i]
    print(video)
    
    mask = os.path.join(path, str(video), 'dino_embeddings','sam2_mask','all_mask.pt')
    mask = torch.load(mask) # N,T,1,H,W
    selected_mask = mask[selected[i]]
    combined_mask = torch.any(selected_mask, dim=0)[None,...]
    
    tmp = combined_mask[0,0,0] # H,W
    H, W = tmp.shape[:2]
    mesh = torch.stack(torch.meshgrid([torch.arange(0,H),torch.arange(0,W)]),dim=-1)
    
    density = 4
    mod_mask = torch.logical_and((mesh[:,:,0] % density)==0,(mesh[:,:,1] % density)==0)
    mask = torch.logical_and(tmp,mod_mask)
    valid_points = torch.nonzero(mask)[...,[1,0]]*torch.tensor([1/W,1/H]).float()
    
    save_dir = os.path.join(path, str(video), 'dino_embeddings','sam2_mask')
    torch.save(combined_mask, os.path.join(save_dir, 'combined_mask.pt'))
    torch.save(valid_points, os.path.join(save_dir, 'queries.pt'))
    print(valid_points.shape)
    # import pdb; pdb.set_trace()
    
    # break