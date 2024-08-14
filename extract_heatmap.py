import argparse
import sys
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from dataset import pklDataset, FeatureDataset

def sample_features(query: torch.Tensor, feature: torch.Tensor):
        # query: (N, xy)
        # feature: (C, H, W)
        normalized_query = (query * 2 - 1)[None,...] # (1, N, xy)
        sampled_features = torch.nn.functional.grid_sample(feature.unsqueeze(0), normalized_query.unsqueeze(0), mode='bilinear', align_corners=True)[0] # (C, 1, N)
        normalized_features = sampled_features / torch.norm(sampled_features, dim=0, keepdim=True)
        return normalized_features

def save_images_as_video(images, output_file):
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, 10.0, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='dataset dir')
    parser.add_argument('--idx', type=int, default=0, help='dataset idx')
    args = parser.parse_args()
    
    image_data = pklDataset(args.data_dir)
    image_data.switch_to(args.idx)
    sample = image_data[0]
    H,W = sample.shape[-3:-1]
    points, occluded = image_data.get_gt()
    valid_points = (torch.from_numpy(points[:,0]))
    occ_mask = torch.from_numpy(occluded[:,0,])
    valid_points = valid_points[occ_mask == False,:].cuda()
    
    dino_out_dir = os.path.join('./davis_dino',f'{args.idx}','dino_embeddings')
    target_feature = 'dino'
    target_file = f'{target_feature}_embed_video.pt'
    heatmap_dir = os.path.join(dino_out_dir,f'{target_feature}_heatmap')
    os.makedirs(heatmap_dir,exist_ok=True)
    
    feature = FeatureDataset(os.path.join(dino_out_dir,target_file))
    
    query_features = sample_features(valid_points,feature[0])[:,0,:].permute(1,0) # (N, C)
    
    cmap = plt.get_cmap('jet')
    
    
    for i in range(query_features.shape[0]):
        point_features = query_features[i]
        heatmaps = []
        for j in tqdm(range(len(feature))):
            frame_features = F.interpolate(feature[j][None,...],size=(H,W),mode='bilinear',align_corners=True)[0] # (C, H, W)
            frame_features = frame_features / torch.norm(frame_features,dim=0,keepdim=True)
            heatmap = (frame_features * point_features[...,None,None]).sum(dim=0).cpu().numpy() # (H, W)
            
            # import pdb; pdb.set_trace()
            heatmap = cmap(heatmap)[:,:,[2,1,0]] # (H, W, 3)
            heatmap = (heatmap * 255.)
            origin_image = image_data[j].astype(np.float64)
            blend_weight = 0.8
            output = heatmap * blend_weight + origin_image * (1 - blend_weight)
            output = output.astype(np.uint8)
            heatmaps.append(output)
        save_images_as_video(heatmaps,os.path.join(heatmap_dir,f'{i}.mp4'))
        # for i in range(query_features.shape[0]):
        #     heatmaps[i].append(output[i])
        # heatmaps.append(output)
    # for i in range(query_features.shape[0]):   
    #     save_images_as_video(heatmaps[i],os.path.join(heatmap_dir,f'{i}.mp4'))
        
            