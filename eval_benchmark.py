import torch
import numpy as np

def eval_tapvid_frame(pred_traj: torch.Tensor = None,pred_vis: torch.Tensor = None,gt_trajectory: torch.Tensor = None,gt_visibility: torch.Tensor = None, frame = 0):
    # pred_traj: (N', T, 2)
    # pred_vis: (N', T)
    # gt_traj: (N, T, 2)
    # gt_vis: (N, T)
    
    gt_traj = gt_trajectory[gt_visibility[:,frame] == True]
    gt_vis = gt_visibility[gt_visibility[:,frame] == True]
    traj_diff = (pred_traj - gt_traj).norm(dim=-1) # (N', T)
    
    eval_frames = [i for i in range(gt_traj.shape[-1]) if i != frame]
    traj_diff = traj_diff[:,eval_frames] # (N', T-1)
    gt_traj = gt_traj[:,eval_frames]
    gt_vis = gt_vis[:,eval_frames]
    
    
    eval_dict = {}
    valid_locs = gt_vis.int().sum()
    pred_valid_locs = pred_vis.int().sum()
    scale = 1/256.
    both_visible = torch.logical_and(pred_vis,gt_vis)
    thres_1 = torch.logical_and(traj_diff < 1*scale,gt_vis).int().sum()
    thres_2 = torch.logical_and(traj_diff < 2*scale,gt_vis).int().sum()
    thres_4 = torch.logical_and(traj_diff < 4*scale,gt_vis).int().sum()
    thres_8 = torch.logical_and(traj_diff < 8*scale,gt_vis).int().sum()
    thres_16 = torch.logical_and(traj_diff < 16*scale,gt_vis).int().sum()
    
    eval_dict['thres_1'] = thres_1.item()
    eval_dict['thres_2'] = thres_2.item()
    eval_dict['thres_4'] = thres_4.item()
    eval_dict['thres_8'] = thres_8.item()
    eval_dict['thres_16'] = thres_16.item()
    eval_dict['gt_visible'] = valid_locs.item()
    eval_dict['thres_1_rate'] = thres_1.item()/valid_locs.item()
    eval_dict['thres_2_rate'] = thres_2.item()/valid_locs.item()
    eval_dict['thres_4_rate'] = thres_4.item()/valid_locs.item()
    eval_dict['thres_8_rate'] = thres_8.item()/valid_locs.item()
    eval_dict['thres_16_rate'] = thres_16.item()/valid_locs.item()
    eval_dict['average_rate'] = (thres_1.item() + thres_2.item() + thres_4.item() + thres_8.item() + thres_16.item())/valid_locs.item()/5
    
    # first n/v is gt, second n/v is pred
    eval_dict['occlusion_correct'] = (pred_vis == gt_vis).int().sum().item()
    eval_dict['occlusion_vv'] = (both_visible).int().sum().item()
    eval_dict['occlusion_vn'] = torch.logical_and(~pred_vis,gt_vis).int().sum().item()
    eval_dict['occlusion_nv'] = torch.logical_and(pred_vis,~gt_vis).int().sum().item()
    eval_dict['occlusion_nn'] = torch.logical_and(~pred_vis,~gt_vis).int().sum().item()
    eval_dict['total_points'] = gt_vis.numel()
    eval_dict['OA'] = eval_dict['occlusion_correct']/eval_dict['total_points']
    
    
    jaccard_1 = torch.logical_and(traj_diff < 1*scale,both_visible).int().sum()
    jaccard_2 = torch.logical_and(traj_diff < 2*scale,both_visible).int().sum()
    jaccard_4 = torch.logical_and(traj_diff < 4*scale,both_visible).int().sum()
    jaccard_8 = torch.logical_and(traj_diff < 8*scale,both_visible).int().sum()
    jaccard_16 = torch.logical_and(traj_diff < 16*scale,both_visible).int().sum()
    
    def jaccard(x):
        return x/(pred_valid_locs+valid_locs-x)
    
    eval_dict['valid_locs'] = valid_locs.item()
    eval_dict['pred_valid_locs'] = pred_valid_locs.item()
    eval_dict['jaccard_1'] = jaccard_1.item()
    eval_dict['jaccard_2'] = jaccard_2.item()
    eval_dict['jaccard_4'] = jaccard_4.item()
    eval_dict['jaccard_8'] = jaccard_8.item()
    eval_dict['jaccard_16'] = jaccard_16.item()
    eval_dict['AJ'] = (jaccard(jaccard_1)+jaccard(jaccard_2)+jaccard(jaccard_4)+jaccard(jaccard_8)+jaccard(jaccard_16))/5
    
    return eval_dict