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
from visualizer import Visualizer
from PIL import Image
from dataset import pklDataset
import json
DEVICE = 'cuda'

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description='',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', help='', action='store_true')
    parser.add_argument('--gpu', help='cuda device') 
    parser.add_argument('--video', help='path to a source video (or a directory with images)', type=Path,
                        default=Path('demo_in/board.mp4'))
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

def run(args):
    config = load_config(args.config)
    logger.info("Loading tracker")
    tracker = config.tracker_class(config)
    logger.info("Tracker loaded")
    initialized = False
    queries = None

    results = []


    if args.data_idx >= 0:
        logger.info("Using data from pkl")
        image_data = pklDataset(args.video)
        image_data.switch_to(args.data_idx)
        curname = image_data.curname()
        sample = image_data[0]
        H,W = sample.shape[-3:-1]
        points, occluded = image_data.get_gt()
        valid_points = (torch.from_numpy(points[:,0,])*torch.Tensor([W-1,H-1]))

    if args.mask:
        logger.info("Using mask")
        density = 4
        mask_path = os.path.dirname(args.video) + '/mask.png'
        mask = torch.from_numpy(np.array(Image.open(mask_path)).astype(np.uint8)[:,:,-1])
        H, W = mask.shape[:2]
        mesh = torch.stack(torch.meshgrid([torch.arange(0,H),torch.arange(0,W)]),dim=-1)
        mod_mask = torch.logical_and((mesh[:,:,0] % density)==0,(mesh[:,:,1] % density)==0)
        mask = torch.logical_and(mask,mod_mask)
        valid_points = torch.nonzero(mask)[...,[1,0]]


    logger.info("Starting tracking")
    
    if args.data_idx >= 0:
        target = image_data
        targetlen = len(target)
    else:
        target = io_utils.get_video_frames(args.video)
        targetlen = io_utils.get_video_length(args.video)
        
    for frame in tqdm(target, total=targetlen):
        if not initialized:
            meta = tracker.init(frame)
            initialized = True
            if args.mask or args.data_idx >= 0:
                queries = valid_points
            else:
                queries = get_queries(frame.shape[:2], args.grid_spacing)
        else:
            meta = tracker.track(frame)

        coords, occlusions = convert_to_point_tracking(meta.result, queries)
        result = meta.result
        result.cpu()
        results.append((result, coords, occlusions))

    edit = None
    if args.edit.exists():
        edit = cv2.imread(str(args.edit), cv2.IMREAD_UNCHANGED)

    logger.info("Drawing the results")
    video_name = args.video.stem
    if args.data_idx >= 0:
        video_name = curname
    video_save_path = args.out 
    vis = Visualizer(video_save_path, pointwidth=1 if args.mask else 2)
    video = []
    traj = []
    occ = []
    
    
    
    
    if args.data_idx >= 0:
        target = image_data
        targetlen = len(target)
    else:
        target = io_utils.get_video_frames(args.video)
        targetlen = io_utils.get_video_length(args.video)
        
    for frame_i, frame in enumerate(tqdm(target, total=targetlen)):
        result, coords, occlusions = results[frame_i]
        video.append(frame)
        traj.append(coords)
        occ.append(occlusions)
        
    video = torch.from_numpy(np.stack(video)).cuda().permute(0,3,1,2)[:,[0,1,2]][None]
    traj = torch.from_numpy(np.stack(traj)).cuda()[None]
    visibility = (torch.from_numpy(np.stack(occ)).cuda()[None][...,None]) < 0.1
    vis.visualize(video, traj, visibility,filename = f'{video_name}_points')
    
    
    if args.data_idx != -1:
        points, occluded = image_data.get_gt()
        valid_points = (torch.from_numpy(points[:,0,[1,0]])*torch.Tensor([H-1,W-1])).long()
        gt_trajectory = (torch.from_numpy(points[...,[1,0]])*torch.Tensor([H-1,W-1])).to(DEVICE)
        gt_visibility = (~torch.from_numpy(occluded)).to(DEVICE)
        trajectory_vis = traj[0,:,:,[1,0]]
        visibility = visibility[0,:,:,0].permute(1,0)
        traj_diff = (trajectory_vis - gt_trajectory.permute(1,0,2)).norm(dim=-1).permute(1,0)
        valid_locs = gt_visibility.int().sum()
        thres_1 = torch.logical_and(traj_diff < 1,gt_visibility).int().sum()
        thres_2 = torch.logical_and(traj_diff < 2,gt_visibility).int().sum()
        thres_4 = torch.logical_and(traj_diff < 4,gt_visibility).int().sum()
        thres_8 = torch.logical_and(traj_diff < 8,gt_visibility).int().sum()
        thres_16 = torch.logical_and(traj_diff < 16,gt_visibility).int().sum()
        occ_mistakes = torch.logical_and(visibility,~gt_visibility).int().sum()
        total_locs = valid_locs + occ_mistakes
        OA = (visibility == gt_visibility).float().mean()
        print("Position accuracy:",thres_1/valid_locs,thres_2/valid_locs,thres_4/valid_locs,thres_8/valid_locs,thres_16/valid_locs)
        print("Average accuracy:", (thres_1+thres_2+thres_4+thres_8+thres_16)/5/valid_locs)
        print("Occlusion accuracy:",OA)
        print("Average Jaccard:", (thres_1+thres_2+thres_4+thres_8+thres_16)/5/total_locs)
    
    if args.data_idx >= 0:
        with open(f'{video_save_path}/results.json','r') as f:
            results_dict = json.load(f)
        results_dict[curname] = {
                                'total':((thres_1+thres_2+thres_4+thres_8+thres_16)/5/valid_locs).item(),
                                '1':(thres_1/valid_locs).item(),
                                '2':(thres_2/valid_locs).item(),
                                '4':(thres_4/valid_locs).item(),
                                '8':(thres_8/valid_locs).item(),
                                '16':(thres_16/valid_locs).item(),
                                'OA':OA.item(),
                                'AJ':((thres_1+thres_2+thres_4+thres_8+thres_16)/5/total_locs).item()
                                }
        results_dict.pop('average',None)
        results_dict['average'] = {
                                    'total':sum([v['total'] for v in results_dict.values()])/len(results_dict),
                                    '1':sum([v['1'] for v in results_dict.values()])/len(results_dict),
                                    '2':sum([v['2'] for v in results_dict.values()])/len(results_dict),
                                    '4':sum([v['4'] for v in results_dict.values()])/len(results_dict),
                                    '8':sum([v['8'] for v in results_dict.values()])/len(results_dict),
                                    '16':sum([v['16'] for v in results_dict.values()])/len(results_dict),
                                    'OA':sum([v['OA'] for v in results_dict.values()])/len(results_dict),
                                    'AJ':sum([v['AJ'] for v in results_dict.values()])/len(results_dict)
                                    }
        with open(f'{video_save_path}/results.json','w') as f:
            json.dump(results_dict,f,indent=4)
    
    
    return 0


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
