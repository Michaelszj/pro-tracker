# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import os
import sys
import argparse
from pathlib import Path
import logging

import numpy as np
import cv2
import torch
from tqdm import tqdm
import einops

from MFT.config import load_config
from MFT.point_tracking import convert_to_point_tracking
import MFT.utils.vis_utils as vu
import MFT.utils.io as io_utils
from MFT.utils.misc import ensure_numpy
from MFT.MFT import MFT
from visualizer import Visualizer
from PIL import Image
from dataset import pklDataset, FeatureDataset, BenchmarkDataset, ImageDataset,resize_tensor
import torchvision.transforms as transforms
from eval_benchmark import eval_tapvid_frame
import json
DEVICE = 'cuda'
benchmark = 'davis'
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--video', help='path to a source video (or a directory with images)', type=str,
                        default='board_new')
    parser.add_argument('--edit', help='path to a RGBA png with a first-frame edit', type=Path,
                        default=Path('demo_in/edit.png'))
    parser.add_argument('--config', help='MFT config file', type=Path, default=Path('configs/MFT_cfg.py'))
    parser.add_argument('--out', help='output directory', type=Path, default=Path('demo_out/'))
    parser.add_argument('--grid_spacing', help='distance between visualized query points', type=int, default=30)
    parser.add_argument('--mask', action='store_true', help='use foreground mask')
    parser.add_argument('--data_idx', type=int, default=-1, help='data index')

    args = parser.parse_args()
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    format = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl, format=format)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return args

def configure_benchmark(image_data: pklDataset, data_idx: int, frame_idx: int, direction: int = 1):
    
    dino_folder = f'./{benchmark}_dino/{data_idx}'
    target = frame_idx
    target = 'mask'
    dino_traj = torch.from_numpy(np.load(os.path.join(dino_folder,f'trajectories/trajectories_{target}.npy'))).to(DEVICE).permute(1, 0, 2)[None,...].to(DEVICE)
    dino_visibility = ~torch.from_numpy(np.load(os.path.join(dino_folder,f'occlusions/occlusion_preds_{target}.npy'))).to(DEVICE).permute(1, 0)[None,...,None]
    image_data.set_start_frame(frame_idx, direction)
    # import pdb; pdb.set_trace()
    print('loading feature and mask')
    # try:
    ours = True
    if ours:
        featdata = FeatureDataset(os.path.join(dino_folder,'dino_embeddings/geo_embed_video.pt'),type='feature')
        # maskdata = FeatureDataset(os.path.join(dino_folder,f'dino_embeddings/sam2_mask/all_mask_{frame_idx}.pt'),type='mask')
        maskdata = FeatureDataset(os.path.join(dino_folder,f'dino_embeddings/sam2_mask/combined_mask.pt'),type='mask')
        featdata.set_start_frame(frame_idx, direction)
        maskdata.set_start_frame(frame_idx, direction)
    else:
        featdata = maskdata = None
    # except:
    # featdata = maskdata = None
    return dino_traj, dino_visibility, featdata, maskdata
    
def track_slides(tracker: MFT, image_data, featdata: FeatureDataset, maskdata: FeatureDataset, 
                 dino_traj, dino_visibility, query):
    
    initialized = False
    # prepare query points
    sample = image_data[0]
    H,W = sample.shape[-3:-1]
    # points, occluded = image_data.get_gt()
    # valid_points = (torch.from_numpy(points[:,0,])*torch.Tensor([W,H]))
    # occ_mask = torch.from_numpy(occluded[:,0,])
    # valid_points = valid_points[occ_mask == False,:]
    
    valid_points = query
    # import pdb; pdb.set_trace()
    # prepare image data
    target = image_data
    targetlen = len(target)
    tracker.targetlen = targetlen
    current_frame = 0
    video = []
    results = []
    
    # track forward
    for frame in tqdm(target, total=targetlen):
        # import pdb; pdb.set_trace()
        video.append(frame)
        if not initialized:
            queries = valid_points
            meta = tracker.init(frame, query = queries, dino_traj = dino_traj, dino_visibility = dino_visibility, featdata=featdata, maskdata=maskdata)
            initialized = True
        else:
            # if current_frame>=80: import pdb; pdb.set_trace()
            meta = tracker.track(frame)

        coords = einops.rearrange(meta.result.flow, 'C H W -> (H W) C')
        occlusions = einops.rearrange(meta.result.occlusion, '1 H W -> (H W)')
        

        result = meta.result
        result.cpu()
        results.append((result, coords, occlusions))
        current_frame += 1
        
    # track backward
    tracker.start_frame_i = targetlen - 1
    tracker.time_direction = -1
    tracker.reverse = True
    current_frame -= 1
    for frame in tqdm(reversed(video[1:-1]), total=targetlen-2):
        current_frame -= 1

        meta = tracker.track(frame)
        coords = einops.rearrange(meta.result.flow, 'C H W -> (H W) C')
        occlusions = einops.rearrange(meta.result.occlusion, '1 H W -> (H W)')
        result = meta.result
        result.cpu()
        results[current_frame] = (result, coords, occlusions)
    
    # save results
    traj = []
    occ = []
    
    for frame_i, frame in enumerate(tqdm(target, total=targetlen)):
        result, coords, occlusions = results[frame_i]
        
        traj.append(coords+queries)
        occ.append(occlusions)
        
    video = torch.from_numpy(np.stack(video)).cuda().permute(0,3,1,2)[:,[0,1,2]][None]
    traj = torch.from_numpy(np.stack(traj)).cuda()[None]
    visibility = (torch.from_numpy(np.stack(occ)).cuda()[None][...,None]) < 0.1
    return video, traj, visibility
    
def run(args):
    config = load_config(args.config)
    logger.info("Loading tracker")
    tracker : MFT = config.tracker_class(config)
    logger.info("Tracker loaded")


    target_path = os.path.join('./casual_video',args.video)
    image_data = ImageDataset(target_path)
    H,W = image_data[0].shape[-3:-1]
    # prepare keypoint
    try:
        # import pdb; pdb.set_trace()
        target = 'grid'
        traj_path = os.path.join(target_path,f'grid_trajectories/{target}_trajectories.npy')
        occ_path = os.path.join(target_path,f'grid_occlusions/{target}_occlusions.npy')
        dino_traj = torch.from_numpy(np.load(traj_path)).to(DEVICE).permute(1, 0, 2)[None,...].to(DEVICE)
        dino_traj = dino_traj*torch.tensor([W/854,H/476]).to(DEVICE)
        # dino_traj = dino_traj*torch.tensor([W/1080,H/720]).to(DEVICE)
        # dino_traj = dino_traj.to(DEVICE)
        dino_visibility = ~torch.from_numpy(np.load(occ_path)).to(DEVICE).permute(1, 0)[None,...,None]
        print('loaded keypoints')
    except:
        dino_traj = dino_visibility = None
        print('no keypoints founded')
    # prepare feature and mask
    try:
        mask_path = os.path.join(target_path,'sam2_mask/all_mask.pt')
        maskdata = FeatureDataset(mask_path,type='mask')
        maskdata.set_start_frame(0,1)
        print('loaded mask')
    except:
        maskdata = None
        print('no mask founded')
    try:
        geo_path = os.path.join(target_path,'dino_embeddings/geo_embed_video.pt')
        featdata = FeatureDataset(geo_path,type='feature')
        featdata.set_start_frame(0,1)
        print('loaded feature')
    except:
        featdata = None
        print('no feature founded')
        
    # prepare query points
    
    query_points = torch.load(os.path.join(target_path,'sam2_mask/query_dense.pt'))*torch.Tensor([W,H])
    video, traj, visibility = track_slides(tracker, image_data, featdata, maskdata, 
                                        dino_traj, dino_visibility, query_points)
    
    # video = torch.from_numpy(image_data.images).permute(0,3,1,2).cuda()[:,[0,1,2]][None]
    # traj = dino_traj
    # visibility = dino_visibility
    traj[0,0] = query_points
    # import pdb; pdb.set_trace()
    
    # traj (1,T,N,2)
    # visibility (1,T,N,1)
    # import pdb; pdb.set_trace()
    
    video_save_path = os.path.join(target_path,'results')
    os.makedirs(video_save_path,exist_ok=True)
    vis = Visualizer(video_save_path, pointwidth=1,linewidth=1,tracks_leave_trace=0,mode='rainbow')
    video_name = args.video
    # import pdb; pdb.set_trace()
    # for f in range(video.shape[1]):
    #     video[0,f] = adjust_saturation(video[0,f], 0.5)
    # video = adjust_contrast(video, 0.5)
    p = -1
    # import pdb; pdb.set_trace()
    if p>=0:
        vis.visualize(video, traj[:,:,p:p+1], visibility[:,:,p:p+1],filename = f'{video_name}_ours')
    else:
        interval = 1
        l = traj.shape[2]
        select = torch.zeros(l,dtype=torch.bool)
        select[::interval] = True
        # visibility[:] = False
        vis.visualize(video, traj[:,:,select], visibility[:,:,select],filename = f'{video_name}_ours')
                    
    return 0

def combine_results(eval_dicts):
    eval_dict = {}
    eval_dict['thres_1'] = sum([v['thres_1'] for v in eval_dicts])
    eval_dict['thres_2'] = sum([v['thres_2'] for v in eval_dicts])
    eval_dict['thres_4'] = sum([v['thres_4'] for v in eval_dicts])
    eval_dict['thres_8'] = sum([v['thres_8'] for v in eval_dicts])
    eval_dict['thres_16'] = sum([v['thres_16'] for v in eval_dicts])
    eval_dict['gt_visible'] = sum([v['gt_visible'] for v in eval_dicts])
    eval_dict['thres_1_rate'] = eval_dict['thres_1']/eval_dict['gt_visible']
    eval_dict['thres_2_rate'] = eval_dict['thres_2']/eval_dict['gt_visible']
    eval_dict['thres_4_rate'] = eval_dict['thres_4']/eval_dict['gt_visible']
    eval_dict['thres_8_rate'] = eval_dict['thres_8']/eval_dict['gt_visible']
    eval_dict['thres_16_rate'] = eval_dict['thres_16']/eval_dict['gt_visible']
    eval_dict['average_rate'] = (eval_dict['thres_1']+eval_dict['thres_2']+eval_dict['thres_4']+eval_dict['thres_8']+eval_dict['thres_16'])/eval_dict['gt_visible']/5
    eval_dict['occlusion_correct'] = sum([v['occlusion_correct'] for v in eval_dicts])
    eval_dict['occlusion_vv'] = sum([v['occlusion_vv'] for v in eval_dicts])
    eval_dict['occlusion_vn'] = sum([v['occlusion_vn'] for v in eval_dicts])
    eval_dict['occlusion_nv'] = sum([v['occlusion_nv'] for v in eval_dicts])
    eval_dict['occlusion_nn'] = sum([v['occlusion_nn'] for v in eval_dicts])
    eval_dict['total_points'] = sum([v['total_points'] for v in eval_dicts])
    eval_dict['OA'] = eval_dict['occlusion_correct']/eval_dict['total_points']
    eval_dict['pred_visible'] = sum([v['pred_visible'] for v in eval_dicts])
    eval_dict['jaccard_1'] = sum([v['jaccard_1'] for v in eval_dicts])
    eval_dict['jaccard_2'] = sum([v['jaccard_2'] for v in eval_dicts])
    eval_dict['jaccard_4'] = sum([v['jaccard_4'] for v in eval_dicts])
    eval_dict['jaccard_8'] = sum([v['jaccard_8'] for v in eval_dicts])
    eval_dict['jaccard_16'] = sum([v['jaccard_16'] for v in eval_dicts])
    def jaccard(x):
        return x/(eval_dict['pred_visible']+eval_dict['gt_visible']-x)
    eval_dict['AJ'] = (jaccard(eval_dict['jaccard_1'])+jaccard(eval_dict['jaccard_2'])+jaccard(eval_dict['jaccard_4'])+jaccard(eval_dict['jaccard_8'])+jaccard(eval_dict['jaccard_16']))/5
    return eval_dict


def save_results(eval_dict, video_save_path, curname):
    save_results_name = f'{video_save_path}/results.json'
    with open(save_results_name,'r') as f:
        results_dict = json.load(f)
    results_dict[curname] = {
                            'total':eval_dict['average_rate'],
                            '1':eval_dict['thres_1_rate'],
                            '2':eval_dict['thres_2_rate'],
                            '4':eval_dict['thres_4_rate'],
                            '8':eval_dict['thres_8_rate'],
                            '16':eval_dict['thres_16_rate'],
                            'occ_vv':eval_dict['occlusion_vv'],
                            'occ_vn':eval_dict['occlusion_vn'],
                            'occ_nv':eval_dict['occlusion_nv'],
                            'occ_nn':eval_dict['occlusion_nn'],
                            'OA':eval_dict['OA'],
                            'AJ':eval_dict['AJ']
                            }
    results_dict.pop('average',None)
    results_dict['average'] = {
                                'total':sum([v['total'] for v in results_dict.values()])/len(results_dict),
                                '1':sum([v['1'] for v in results_dict.values()])/len(results_dict),
                                '2':sum([v['2'] for v in results_dict.values()])/len(results_dict),
                                '4':sum([v['4'] for v in results_dict.values()])/len(results_dict),
                                '8':sum([v['8'] for v in results_dict.values()])/len(results_dict),
                                '16':sum([v['16'] for v in results_dict.values()])/len(results_dict),
                                'occ_vv':sum([v['occ_vv'] for v in results_dict.values() if v.get('occ_vv') is not None]),
                                'occ_vn':sum([v['occ_vn'] for v in results_dict.values() if v.get('occ_vn') is not None]),
                                'occ_nv':sum([v['occ_nv'] for v in results_dict.values() if v.get('occ_nv') is not None]),
                                'occ_nn':sum([v['occ_nn'] for v in results_dict.values() if v.get('occ_nn') is not None]),
                                'OA':sum([v['OA'] for v in results_dict.values()])/len(results_dict),
                                'AJ':sum([v['AJ'] for v in results_dict.values()])/len(results_dict)
                                }
    with open(save_results_name,'w') as f:
        json.dump(results_dict,f,indent=4)
        
        
def adjust_contrast(img, contrast_factor):
    mean = torch.mean(img.float())  # 计算图像的均值
    # 应用对比度调整公式
    img_adjusted = (img - mean) * contrast_factor + mean
    return img_adjusted.int()

def adjust_saturation(img, saturation_factor):
    img = img.float()
    # 计算每个像素的亮度值，按照 NTSC 的标准加权求和
    gray = 0.299 * img[0, :, :] + 0.587 * img[1, :, :] + 0.114 * img[2, :, :]
    gray = gray.unsqueeze(0).expand_as(img)  # 扩展维度以匹配原图像的形状
    # 将灰度图像与原始图像混合
    img_adjusted = (img - gray) * saturation_factor + gray
    return img_adjusted.int()
        
def get_queries(frame_shape, spacing):
    H, W = frame_shape
    xs = np.arange(0, W, spacing)
    ys = np.arange(0, H, spacing)

    xs, ys = np.meshgrid(xs, ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    queries = np.vstack((flat_xs, flat_ys)).T
    return torch.from_numpy(queries).float().cuda()

def draw_dots(frame, coords, occlusions):
    canvas = frame.copy()
    N = coords.shape[0]

    for i in range(N):
        occl = occlusions[i] > 0.5
        if not occl:
            thickness = 1 if occl else -1
            vu.circle(canvas, coords[i, :], radius=3, color=vu.RED, thickness=thickness)

    return canvas

def draw_edit(frame, result, edit):
    occlusion_in_template = result.occlusion
    template_visible_mask = einops.rearrange(occlusion_in_template, '1 H W -> H W') < 0.5
    template_visible_mask = template_visible_mask.cpu()
    edit_mask = torch.from_numpy(edit[:, :, 3] > 0)
    template_visible_mask = torch.logical_and(template_visible_mask, edit_mask)

    edit_alpha = einops.rearrange(edit[:, :, 3], 'H W -> H W 1').astype(np.float32) / 255.0
    premult = edit[:, :, :3].astype(np.float32) * edit_alpha
    color_transfer = ensure_numpy(result.warp_forward(premult, mask=template_visible_mask))
    color_transfer = np.clip(color_transfer, 0, 255).astype(np.uint8)
    alpha_transfer = ensure_numpy(result.warp_forward(
        einops.rearrange(edit[:, :, 3], 'H W -> H W 1'),
        mask=template_visible_mask
    ))
    vis = vu.blend_with_alpha_premult(color_transfer, vu.to_gray_3ch(frame), alpha_transfer)
    return vis

from ipdb import iex
@iex
def main():
    args = parse_arguments()
    return run(args)


if __name__ == '__main__':
    results = main()
