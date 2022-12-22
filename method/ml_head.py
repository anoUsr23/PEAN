from turtle import forward
import torch
import torch.nn as nn
import numpy as np

from utils.model_utils import get_gauss_props_from_clip_indices, generate_gauss_weight
import torch.nn.functional as F

class ML_Head(nn.Module):
    def __init__(self, config):
        super(ML_Head, self).__init__()
        self.config = config
        self.num_props = config.num_props
        self.sigma = config.sigma # cpl

        # infer
        self.num_gauss_center = config.num_gauss_center
        self.num_gauss_width = config.num_gauss_width


    def get_ml_results(self, key_clip_indices, query_labels, video_feat, video_feat_mask, modular_roberta_feat, epoch=1, train=True):

        num_video, max_frames, _ = video_feat.shape
        video_feat = video_feat[query_labels]
        video_feat_mask = video_feat_mask[query_labels]

        num_query, d = modular_roberta_feat.shape

        ###########################
        # generate gaussian mask
        key_clip_center_prop, _ = get_gauss_props_from_clip_indices(key_clip_indices, self.num_gauss_center, self.num_gauss_width) # [nq, nv]
        selected_center_prop = key_clip_center_prop[[i for i in range(key_clip_center_prop.shape[0])], query_labels] # [nq, ] -- only matched
        gauss_center = selected_center_prop.unsqueeze(-1).expand(num_query, self.num_props).reshape(-1)                                                                         
        gauss_width = torch.linspace(0.05, 1, steps=self.num_props).unsqueeze(0).expand(num_query, -1).reshape(-1).to(video_feat.device)


        # positive proposal
        gauss_weight_l = generate_gauss_weight(max_frames, gauss_center, gauss_width, sigma=self.sigma)
        props_sim_scores = self.gauss_guided_q2v_similarity(gauss_weight_l, modular_roberta_feat, video_feat, self.num_props) # gauss_weighted pooling
        gauss_guided_q2vscore, ggq2v_indices = props_sim_scores.max(dim=-1)

        return {
            'width': gauss_width,
            'center': gauss_center,
            'gauss_guided_q2vscore': gauss_guided_q2vscore,
            'props_sim_scores': props_sim_scores,
        }

    ########################
    # inference
    def get_moment_level_inference_results(self, video_feat, video_feat_mask, modular_roberta_feat):
        num_video, max_ctx_len, _ = video_feat.shape
        # generate proposal
        gauss_center = torch.linspace(0, 1, steps=self.num_gauss_center)
        gauss_center = gauss_center.unsqueeze(-1).expand(-1, self.num_gauss_width).reshape(-1)
        gauss_width = torch.linspace(0.05, 1, steps=self.num_gauss_width).unsqueeze(0).expand(self.num_gauss_center, -1).reshape(-1)

        gauss_center = gauss_center.to(video_feat.device)
        gauss_width = gauss_width.to(video_feat.device)

        # calc similarity score, using gaussian weight guided pooling
        gauss_weight = generate_gauss_weight(max_ctx_len, gauss_center, gauss_width, sigma=self.sigma)
        props_sim_scores = []
        # normalize query feat
        modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1)
        for weight in gauss_weight:
            video_gauss_weight = weight.unsqueeze(0).expand(num_video, -1)
            gauss_guided_global_vid_feat = self.gauss_weighted_pooling(video_feat, video_feat_mask, video_gauss_weight, self.num_props)
            gauss_guided_global_vid_feat = F.normalize(gauss_guided_global_vid_feat, dim=-1) # normalize
            _sim_scores = torch.matmul(modular_roberta_feat, gauss_guided_global_vid_feat.transpose(0,1))
            props_sim_scores.append(_sim_scores.unsqueeze(-1))
        sim_scores = torch.cat(props_sim_scores, dim=-1).max(dim=-1)[0]

        return sim_scores


    def gauss_guided_q2v_similarity(self, gauss_weight, modular_roberta_feat, video_feat, num_props):
        
        num_video, _, _ = video_feat.shape
        global_props_vid_feat = self.gauss_weighted_pooling(video_feat, None, gauss_weight, num_props).view(num_video, num_props, -1)

        # add normalization
        modular_roberta_feat = F.normalize(modular_roberta_feat, dim=-1)
        global_props_vid_feat = F.normalize(global_props_vid_feat, dim=-1)

        props_sim_scores = torch.einsum("nd,mpd->nmp", modular_roberta_feat, global_props_vid_feat)

        return props_sim_scores
    
    def gauss_weighted_pooling(self, frame_feat, frame_mask, gauss_weight, num_props):
        nv, lv, d = frame_feat.shape
        if frame_feat.shape[0] != gauss_weight.shape[0]:
            frame_feat = frame_feat.unsqueeze(1).expand(nv, num_props, lv, d).reshape(nv*num_props, lv, d)
        gauss_weight = (gauss_weight + 1e-10) / gauss_weight.sum(dim=-1, keepdim=True)# normalize
        global_props_vid_feat = torch.bmm(gauss_weight.unsqueeze(1), frame_feat).squeeze(1)
        return global_props_vid_feat
    


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)