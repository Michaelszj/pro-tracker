# -*- origami-fold-style: triple-braces; coding: utf-8; -*-
import einops
import numpy as np
import torch
from types import SimpleNamespace
import logging
from MFT.results import FlowOUTrackingResult
from MFT.utils.timing import general_time_measurer
from dataset import FeatureDataset
logger = logging.getLogger(__name__)


class MFT():
    def __init__(self, config):
        """Create MFT tracker
        args:
          config: a MFT.config.Config, for example from configs/MFT_cfg.py"""
        self.C = config   # must be named self.C, will be monkeypatched!
        self.flower = config.flow_config.of_class(config.flow_config)  # init the OF
        self.device = 'cuda'

    def init(self, img, start_frame_i=0, time_direction=1, flow_cache=None, query : torch.Tensor = None, 
             dino_traj: torch.Tensor = None, dino_visibility: torch.Tensor = None, featdata: FeatureDataset = None, 
            maskdata: FeatureDataset = None,**kwargs):
        """Initialize MFT on first frame

        args:
          img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          start_frame_i: [optional] init frame number (used for caching)
          time_direction: [optional] forward = +1, or backward = -1 (used for caching)
          flow_cache: [optional] MFT.utils.io.FlowCache (for caching OF on GPU, RAM, or SSD)
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: initial frame result container, with initial (zero-motion) MFT.results.FlowOUTrackingResult in meta.result 
        """
        self.img_H, self.img_W = img.shape[:2]
        self.start_frame_i = start_frame_i
        self.current_frame_i = self.start_frame_i
        assert time_direction in [+1, -1]
        self.time_direction = time_direction
        self.flow_cache = flow_cache
        self.targetlen = 0
        self.reverse = False
        self.memory = {
            self.start_frame_i: {
                'img': img,
                'result': FlowOUTrackingResult.identity((1, query.shape[0]), device=self.device)
            }
        }
        
        self.template_img = img.copy()
        self.dino_traj = dino_traj
        self.dino_visibility = dino_visibility
        self.featdata = featdata
        self.maskdata = maskdata

        self.query = query.permute(1, 0).unsqueeze(1).to(self.device) # (xy, 1, N)
        self.query_features = self.sample_features(self.query)
        
        meta = SimpleNamespace()
        meta.result = self.memory[self.start_frame_i]['result'].clone().cpu()
        return meta
    
    def sample_features(self, query: torch.Tensor):
        # query: (xy, 1, N)
        normalized_query = (query.permute(1,2,0) / torch.Tensor([self.img_W, self.img_H]).to(self.device)) * 2 - 1 # (1, N, xy)
        featmap = self.featdata[self.current_frame_i] # (C, H, W)
        sampled_features = torch.nn.functional.grid_sample(featmap.unsqueeze(0), normalized_query.unsqueeze(0), mode='bilinear', align_corners=True)[0] # (C, 1, N)
        
        normalized_features = sampled_features  / torch.norm(sampled_features, dim=0, keepdim=True)
        return normalized_features
    
    def sample_masks(self, query: torch.Tensor):
        # query: (xy, 1, N)
        normalized_query = (query.permute(1,2,0) / torch.Tensor([self.img_W, self.img_H]).to(self.device)) * 2 - 1 # (1, N, xy)
        featmap = self.maskdata[self.current_frame_i] # (C, H, W)
        sampled_features = torch.nn.functional.grid_sample(featmap.unsqueeze(0), normalized_query.unsqueeze(0), mode='bilinear', align_corners=True)[0] # (C, 1, N)
        normalized_features = sampled_features
        
        return normalized_features
        
        
    def track(self, input_img, debug=False, **kwargs):
        """Track one frame

        args:
          input_img: opencv image (numpy uint8 HxWxC array with B, G, R channel order)
          debug: [optional] enable debug visualizations
          kwargs: [unused] - for compatibility with other trackers

        returns:
          meta: current frame result container, with MFT.results.FlowOUTrackingResult in meta.result
                The meta.result represents the accumulated flow field from the init frame, to the current frame
        """
        meta = SimpleNamespace()
        self.current_frame_i += self.time_direction

        # OF(init, t) candidates using different deltas
        delta_results = {}
        already_used_left_ids = []
        chain_timer = general_time_measurer('chain', cuda_sync=True, start_now=False, active=self.C.timers_enabled)
        for delta in self.C.deltas:
            # candidates are chained from previous result (init -> t-delta) and flow (t-delta -> t)
            # when tracking backward, the chain consists of previous result (init -> t+delta) and flow(t+delta -> t)
            left_id = self.current_frame_i - delta * self.time_direction
            right_id = self.current_frame_i

            # we must ensure that left_id is not behind the init frame
            if self.is_before_start(left_id):
                if np.isinf(delta):
                    left_id = self.start_frame_i
                else:
                    continue
            left_id = int(left_id)

            # because of this, different deltas can result in the same left_id, right_id combination
            # let's not recompute the same candidate multiple times
            if left_id in already_used_left_ids:
                continue

            left_img = self.memory[left_id]['img']
            right_img = input_img

            template_to_left = self.memory[left_id]['result']

            flow_init = None
            use_cache = np.isfinite(delta) or self.C.cache_delta_infinity
            left_to_right = get_flowou_with_cache(self.flower, left_img, right_img, flow_init,
                                                  self.flow_cache, left_id, right_id,
                                                  read_cache=use_cache, write_cache=use_cache)

            chain_timer.start()
            delta_results[delta] = chain_results(template_to_left, left_to_right, self.query)
            already_used_left_ids.append(left_id)
            chain_timer.stop()

        chain_timer.report('mean')
        chain_timer.report('sum')

        selection_timer = general_time_measurer('selection', cuda_sync=True, start_now=True,
                                                active=self.C.timers_enabled)
        used_deltas = sorted(list(delta_results.keys()), key=lambda delta: 0 if np.isinf(delta) else delta)
        all_results = [delta_results[delta] for delta in used_deltas]
        # if self.reverse:
        #     all_results.append(self.memory[self.current_frame_i]['result'])
        all_flows = torch.stack([result.flow for result in all_results], dim=0)  # (N_delta, xy, H, W)
        all_sigmas = torch.stack([result.sigma for result in all_results], dim=0)  # (N_delta, 1, H, W)
        all_occlusions = torch.stack([result.occlusion for result in all_results], dim=0)  # (N_delta, 1, H, W)
        
        dino_flow = (self.dino_traj[0,self.current_frame_i]-self.dino_traj[0,0]).permute(1, 0).unsqueeze(1).to(self.device) # (xy, 1, N)
        dino_occlusion = (~self.dino_visibility[0,self.current_frame_i].permute(1, 0).unsqueeze(1).to(self.device)).float() # (1, 1, N)
        # all_occlusions[all_sigmas > 1] = 1
        # import pdb; pdb.set_trace()
        input_queries = (torch.cat([all_flows,dino_flow[None,...]]) + self.query).permute(2,1,0,3)[0] # (xy, N_delta, N)
        
        
        # if self.featdata is not None:
        #     sampled_features = self.sample_features(input_queries) # (C, N_delta+1, N)
        #     tracker2_feature = sampled_features[:,-1:] # (C, 1, N)
        #     # sampled_features = sampled_features[:,:-1] # (C, N_delta, N)
        #     similarities = (sampled_features * self.query_features).sum(dim=0)[:,None,None,:] # (N_delta, 1, H, W)
        #     similarity_threshold = 0.5
        #     # all_occlusions[similarities[:-1] < similarity_threshold] = 1
        #     # dino_occlusion[similarities[-1] < similarity_threshold] = 1
            
            
        if self.maskdata is not None:
            sampled_masks = self.sample_masks(input_queries) # (C, N_delta+1, N)
            tracker2_mask = sampled_masks[:,-1:] # (C, 1, N)
            sampled_masks = sampled_masks[:,:-1] # (C, N_delta, N)
            point_map = self.maskdata.point_map # (N,)
            # import pdb; pdb.set_trace()
            sampled_masks = sampled_masks[point_map, torch.arange(sampled_masks.shape[1]).unsqueeze(1), torch.arange(sampled_masks.shape[2]).unsqueeze(0)]
            # sampled_masks = sampled_masks.diagonal(dim1=0,dim2=2) # (N_delta, N)
            mask_thres = 0.01
            all_occlusions[sampled_masks[:,None,None,:] < mask_thres] = 1
            
            # tracker2_feature = tracker2_feature.diagonal(dim1=0,dim2=2) # (1, N)
            # dino_occlusion[tracker2_feature[None,...] < mask_thres] = 1
            
        
        
        scores = -all_sigmas
        scores[all_occlusions > self.C.occlusion_threshold] = -float('inf')

        best = scores.max(dim=0, keepdim=True)
        selected_delta_i = best.indices  # (1, 1, H, W)

        best_flow = all_flows.gather(dim=0,
                                     index=einops.repeat(selected_delta_i,
                                                         'N_delta 1 H W -> N_delta xy H W',
                                                         xy=2))
        best_occlusions = all_occlusions.gather(dim=0, index=selected_delta_i)
        best_sigmas = all_sigmas.gather(dim=0, index=selected_delta_i)
        selected_flow, selected_occlusion, selected_sigmas = best_flow, best_occlusions, best_sigmas

        selected_flow = einops.rearrange(selected_flow, '1 xy H W -> xy H W', xy=2)
        selected_occlusion = einops.rearrange(selected_occlusion, '1 1 H W -> 1 H W')
        selected_sigmas = einops.rearrange(selected_sigmas, '1 1 H W -> 1 H W')
        
        flow_distance = torch.sum(torch.square(all_flows - selected_flow), dim=1) # (N_delta, H, W)
        flow_filter_threshold = 2
        all_occlusions[:,0][flow_distance > flow_filter_threshold] = 1
        

        occluded = all_occlusions > self.C.occlusion_threshold
        weight = 1/torch.square(all_sigmas)
        weight[occluded] = 0
        sum_flow = torch.sum(all_flows * weight, dim=0)
        sum_weight = torch.sum(weight, dim=0)
        average_flow = sum_flow
        average_flow[:,sum_weight[0] > 0] /= sum_weight[sum_weight > 0] # (xy, H, W)
        new_sigma = torch.zeros_like(sum_weight) # (1, H, W)
        new_sigma[sum_weight > 0] = 1/torch.sqrt(sum_weight[sum_weight > 0]) # (1, H, W)
        new_occlusion = (~(sum_weight > 0)).float() # (1, H, W)
        # import pdb;pdb.set_trace()
        if self.reverse:
            prev_result : FlowOUTrackingResult = self.memory[self.current_frame_i]['result']
            prev_flow = prev_result.flow
            prev_sigma = prev_result.sigma
            prev_occlusion = prev_result.occlusion
            change_mask = torch.logical_and(prev_occlusion[0] >= 1.-self.C.occlusion_threshold, new_occlusion[0] <= self.C.occlusion_threshold)
            prev_flow[:,change_mask] = average_flow[:,change_mask]
            prev_sigma[0,change_mask] = selected_sigmas[0,change_mask]
            prev_occlusion[0,change_mask] = 0
            # if self.current_frame_i == 30:
            #     import pdb; pdb.set_trace()
            average_flow = prev_flow
            selected_sigmas = prev_sigma
            new_sigma = prev_sigma
            new_occlusion = prev_occlusion
        
        
        
        # if (selected_flow-average_flow).abs().max() > 1e-3:
        #     import pdb; pdb.set_trace()
        
        # result = FlowOUTrackingResult(selected_flow, selected_occlusion, selected_sigmas)
        
        
        
        # import pdb; pdb.set_trace()
        last_flow = self.memory[self.current_frame_i-self.time_direction]['result'].flow
        last_occlusion = self.memory[self.current_frame_i-self.time_direction]['result'].occlusion
        flow_consistency_threshold = 0
        flow_consistency_mask = torch.logical_and(torch.sum(torch.square(last_flow - dino_flow), dim=0, keepdim=True) > flow_consistency_threshold, last_occlusion < self.C.occlusion_threshold)

        replace_mask = torch.logical_and(dino_occlusion[0] == 0, new_occlusion[0] == 1).squeeze()
        flow_mask = (new_occlusion[0] == 1).squeeze()
        average_flow[:,:,flow_mask] = dino_flow[:,:,flow_mask]
        selected_sigmas[:,:,replace_mask] = 0
        new_occlusion[:,:,replace_mask] = 0
        result = FlowOUTrackingResult(average_flow, new_occlusion, selected_sigmas)
        # replace_mask = torch.logical_and(dino_occlusion[0] == 1, new_occlusion[0] == 0).squeeze()
        # dino_flow[:,:,replace_mask] = average_flow[:,:,replace_mask]
        # dino_occlusion[:,:,replace_mask] = 0
        # dino_sigmas = torch.ones_like(selected_sigmas).to(self.device)
        # dino_sigmas[:,:,replace_mask] = selected_sigmas[:,:,replace_mask]
        # result = FlowOUTrackingResult(dino_flow, dino_occlusion, dino_sigmas)

        # mark flows pointing outside of the current image as occluded
        invalid_mask = einops.rearrange(result.invalid_mask(self.img_H,self.img_W,self.query), 'H W -> 1 H W')
        result.occlusion[invalid_mask] = 1
        selection_timer.report()

        out_result = result.clone()
            
        meta.result = out_result
        meta.result.cpu()

        self.memory[self.current_frame_i] = {'img': input_img,
                                             'result': result}

        self.cleanup_memory()
        return meta

    # @profile
    def cleanup_memory(self):
        # max delta, ignoring the inf special case
        try:
            max_delta = np.amax(np.array(self.C.deltas)[np.isfinite(self.C.deltas)])
        except ValueError:  # only direct flow
            max_delta = 0
        has_direct_flow = np.any(np.isinf(self.C.deltas))
        memory_frames = list(self.memory.keys())
        for mem_frame_i in memory_frames:
            if mem_frame_i == self.start_frame_i and has_direct_flow:
                continue

            if self.time_direction > 0 and mem_frame_i + max_delta > self.current_frame_i:
                # time direction     ------------>
                # mem_frame_i ........ current_frame_i ........ (mem_frame_i + max_delta)
                # ... will be needed later
                continue

            if self.time_direction < 0 and mem_frame_i - max_delta < self.current_frame_i:
                # time direction     <------------
                # (mem_frame_i - max_delta) ........ current_frame_i .......... mem_frame_i
                # ... will be needed later
                continue

            # del self.memory[mem_frame_i]

    def is_before_start(self, frame_i):
        return ((self.time_direction > 0 and frame_i < self.start_frame_i) or  # forward
                (self.time_direction < 0 and frame_i > self.start_frame_i))    # backward


# @profile
def get_flowou_with_cache(flower, left_img, right_img, flow_init=None,
                          cache=None, left_id=None, right_id=None,
                          read_cache=False, write_cache=False):
    """Compute flow from left_img to right_img. Possibly with caching.

    args:
        flower: flow wrapper
        left_img: (H, W, 3) BGR np.uint8 image
        right_img: (H, W, 3) BGR np.uint8 image
        flow_init: [optional] (2, H, W) tensor with flow initialisation (caching is disabled when flow_init used)
        cache: [optional] cache object with
        left_id: [optional] frame number of left_img
        right_id: [optional] frame number of right_img
        read_cache: [optional] enable loading from flow cache
        write_cache: [optional] enable writing into flow cache

    returns:
        flowou: FlowOUTrackingResult
    """
    must_compute = not read_cache
    if read_cache and flow_init is None:
        # attempt loading cached flow
        assert left_id is not None
        assert right_id is not None

        try:
            assert cache is not None
            flow_left_to_right, occlusions, sigmas = cache.read(left_id, right_id)
            assert flow_left_to_right is not None
        except Exception:
            must_compute = True

    if must_compute:  # read_cache == False, flow not cached yet, or some cache read error
        # print(f'computing flow {left_id}->{right_id}')
        flow_left_to_right, extra = flower.compute_flow(left_img, right_img, mode='flow',
                                                        init_flow=flow_init)
        occlusions, sigmas = extra['occlusion'], extra['sigma']

    if (cache is not None) and write_cache and must_compute and (flow_init is None):
        cache.write(left_id, right_id, flow_left_to_right, occlusions, sigmas)
    flowou = FlowOUTrackingResult(flow_left_to_right, occlusions, sigmas)
    return flowou


def chain_results(left_result: FlowOUTrackingResult, right_result: FlowOUTrackingResult, query : torch.Tensor):
    flow = left_result.chain(right_result.flow, query)
    occlusions = torch.maximum(left_result.occlusion,
                               left_result.warp_backward(right_result.occlusion, query))
    new_sigma = left_result.warp_backward(right_result.sigma, query)
    sigmas = torch.sqrt(torch.square(left_result.sigma) +
                        torch.square(new_sigma))
    sigma_threshold = 2.
    # occlusions[new_sigma>sigma_threshold] = 1
    return FlowOUTrackingResult(flow, occlusions, sigmas)
