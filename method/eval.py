import numpy as np
import logging
import torch.backends.cudnn as cudnn
import os
import pickle
from method.model import PEAN
from torch.utils.data import DataLoader
from method.data_provider import Dataset4Training,VisDataSet,\
    TxtDataSet,read_video_ids, collate_frame_val, collate_text_val
from tqdm import tqdm
from collections import defaultdict
import torch
from utils.basic_utils import AverageMeter, BigFile, read_dict
from method.config import TestOptions
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def ap_score(sorted_labels):
    nr_relevant = len([x for x in sorted_labels if x > 0])
    if nr_relevant == 0:
        return 0.0

    length = len(sorted_labels)
    ap = 0.0
    rel = 0

    for i in range(length):
        lab = sorted_labels[i]
        if lab >= 1:
            rel += 1
            ap += float(rel) / (i + 1.0)
    ap /= nr_relevant
    return ap

def get_gt(video_metas, query_metas):
    v2t_gt = []
    for vid_id in video_metas:
        v2t_gt.append([])
        for i, query_id in enumerate(query_metas):
            if query_id.split('#', 1)[0] == vid_id:
                v2t_gt[-1].append(i)

    t2v_gt = {}
    for i, t_gts in enumerate(v2t_gt):
        for t_gt in t_gts:
            t2v_gt.setdefault(t_gt, [])
            t2v_gt[t_gt].append(i)

    return v2t_gt, t2v_gt

def eval_q2m(indices, q2m_gts):
    n_q, n_m = indices.shape

    gt_ranks = np.zeros((n_q,), np.int32)
    aps = np.zeros(n_q)
    for i in range(n_q):
        sorted_idxs = indices[i]
        # sorted_idxs = np.argsort(s)
        rank = n_m + 1
        tmp_set = []
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0] + 1
            if tmp < rank:
                rank = tmp

        gt_ranks[i] = rank

    # compute metrics
    r1 = 100.0 * len(np.where(gt_ranks <= 1)[0]) / n_q
    r5 = 100.0 * len(np.where(gt_ranks <= 5)[0]) / n_q
    r10 = 100.0 * len(np.where(gt_ranks <= 10)[0]) / n_q
    r100 = 100.0 * len(np.where(gt_ranks <= 100)[0]) / n_q
    medr = np.median(gt_ranks)
    meanr = gt_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def eval_q2m_v2(scores, q2m_gts):
    n_q, n_m = scores.shape
    sorted_indices = np.argsort(scores)
    
    gt_list = []
    for i in sorted(q2m_gts):
        gt_list.append(q2m_gts[i][0])
    gt_list = np.array(gt_list)
    pred_ranks = np.argwhere(sorted_indices==gt_list[:, np.newaxis])[:, 1]

    r1 = 100 * (pred_ranks==0).sum() / n_q
    r5 = 100 * (pred_ranks<5).sum() / n_q
    r10 = 100 * (pred_ranks<10).sum() / n_q
    r100 = 100 * (pred_ranks<100).sum() / n_q
    medr = np.median(pred_ranks)
    meanr = pred_ranks.mean()

    return (r1, r5, r10, r100, medr, meanr)

def t2v_map(c2i, t2v_gts):
    perf_list = []
    for i in range(c2i.shape[0]):
        d_i = c2i[i, :]
        labels = [0]*len(d_i)

        x = t2v_gts[i][0]
        labels[x] = 1

        sorted_labels = [labels[x] for x in d_i]

        current_score = ap_score(sorted_labels)
        perf_list.append(current_score)
    return np.mean(perf_list)


def computing_q2v_info(model, eval_text_dataset, eval_video_dataset, opt):
    model.eval()

    query_eval_loader = DataLoader(eval_text_dataset, collate_fn=collate_text_val, batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)
    
    context_dataloader = DataLoader(eval_video_dataset, collate_fn=collate_frame_val, batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers, shuffle=False, pin_memory=opt.pin_memory)

    query_metas = []
    video_metas = []
    video_meta_flag = False
    clip_scale_scores = []
    frame_scale_scores = []
    score_sum = []

    for i, query_batch in tqdm(enumerate(query_eval_loader), desc="Computing query2video scores", total=len(query_eval_loader)):
        query_metas.extend(query_batch[-1])
        query_feat = query_batch[0].to(opt.device)
        query_mask = query_batch[1].to(opt.device)
        # 
        _clip_scale_scores = []
        _frame_scale_scores = []
        _score_sum = []

        print("computing q2v similarity for video batches")
        for j, video_batch in tqdm(enumerate(context_dataloader), desc="video_batches", total=len(context_dataloader)):
            if not video_meta_flag:
                video_metas.extend(video_batch[-1])
            batch_frame_video_feat = video_batch[0].to(opt.device)
            batch_frame_mask = video_batch[1].to(opt.device)

            batch_video_feat, batch_video_proposal_feat, batch_frame_mask= model.encode_context(batch_frame_video_feat, batch_frame_mask)

            batch_clip_scale_scores, batch_atten_based_scores, var4ml = model.get_pred_from_raw_query(
                query_feat, query_mask, None, batch_video_proposal_feat, batch_video_feat, batch_frame_mask)
            
            batch_frame_scale_scores = model.ml_head.get_moment_level_inference_results(batch_video_feat, batch_frame_mask, var4ml['video_query'])

            # scores
            batch_score_sum = opt.clip_scale_w*batch_clip_scale_scores + opt.frame_scale_w*batch_frame_scale_scores
            _clip_scale_scores.append(batch_clip_scale_scores)
            _frame_scale_scores.append(batch_frame_scale_scores)
            _score_sum.append(batch_score_sum)
        video_meta_flag = True
        
        _clip_scale_scores = torch.cat(_clip_scale_scores, dim=1)
        _frame_scale_scores = torch.cat(_frame_scale_scores, dim=1)

        _score_sum = torch.cat(_score_sum, dim=1)

        clip_scale_scores.append(_clip_scale_scores)
        frame_scale_scores.append(_frame_scale_scores)
        score_sum.append(_score_sum)

    clip_scale_scores = torch.cat(clip_scale_scores, dim=0)
    clip_sorted_indices = torch.argsort(clip_scale_scores, dim=1, descending=True).cpu().numpy().copy()
    frame_scale_scores = torch.cat(frame_scale_scores, dim=0)
    frame_sorted_indices = torch.argsort(frame_scale_scores, dim=1, descending=True).cpu().numpy().copy()
    score_sum = torch.cat(score_sum, dim=0)
    sum_sorted_indices = torch.argsort(score_sum, dim=1, descending=True).cpu().numpy().copy()

    return clip_sorted_indices, frame_sorted_indices, sum_sorted_indices, query_metas, video_metas


def get_perf(t2v_sorted_indices, t2v_gt):

    # video retrieval
    (t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr) = eval_q2m(t2v_sorted_indices, t2v_gt)
    t2v_map_score = t2v_map(t2v_sorted_indices, t2v_gt)


    logging.info(" * Text to Video:")
    logging.info(" * r_1_5_10_100, medr, meanr: {}".format([round(t2v_r1, 1), round(t2v_r5, 1), round(t2v_r10, 1), round(t2v_r100, 1)]))
    logging.info(" * recall sum: {}".format(round(t2v_r1+t2v_r5+t2v_r10+t2v_r100, 1)))
    logging.info(" * mAP: {}".format(round(t2v_map_score, 4)))
    logging.info(" * "+'-'*10)

    return (t2v_r1, t2v_r5, t2v_r10,t2v_r100, t2v_medr, t2v_meanr, t2v_map_score)


def eval_epoch(model, val_video_dataset, val_text_dataset, opt, epoch=999):
    model.eval()
    logger.info("*"*60)
    logger.info('*'*20 + f" Eval epoch: {epoch}" + '*'*20)

    clip_sorted_indices, frame_sorted_indices, sum_sorted_indices, query_metas, video_metas = computing_q2v_info(model,
                                                                                                        val_text_dataset,
                                                                                                        val_video_dataset,
                                                                                                        opt
                                                                                                        )

    v2t_gt, t2v_gt = get_gt(video_metas, query_metas)
    logger.info('clip_scale_scores:')
    get_perf(clip_sorted_indices, t2v_gt)
    logger.info('frame_scale_scores:')
    get_perf(frame_sorted_indices, t2v_gt)
    logger.info('score_sum:')
    t2v_r1, t2v_r5, t2v_r10, t2v_r100, t2v_medr, t2v_meanr, t2v_map_score = get_perf(sum_sorted_indices, t2v_gt)
    currscore = 0
    currscore += (t2v_r1 + t2v_r5 + t2v_r10 + t2v_r100)

    return currscore


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    ckpt_filepath = os.path.join(opt.ckpt_filepath)
    checkpoint = torch.load(ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    NAME_TO_MODELS = {'PEAN':PEAN}
    model = NAME_TO_MODELS[opt.model_name](loaded_model_cfg)
    
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(model, device_ids=opt.device_ids)  # use multi GPU
    return model

def start_inference(opt=None):
    logger.info("Setup config, data and model...")
    if opt is None:
        opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    rootpath = opt.root_path
    collection = opt.collection
    testCollection = '%stest' % collection

    cap_file = {'test': '%s.caption.txt' % testCollection}

    # caption
    caption_files = {x: os.path.join(rootpath, collection, 'TextData', cap_file[x])
                     for x in cap_file}

    text_feat_path = os.path.join(rootpath, collection, 'TextData', 'roberta_%s_query_feat.hdf5' % collection)
    
    video_feat_path = os.path.join(rootpath, collection, '%s_%s.hdf5'%(collection, opt.visual_feature))

    test_video_ids_list = read_video_ids(caption_files['test'])
    test_vid_dataset = VisDataSet(video_feat_path, opt,
                                               video_ids=test_video_ids_list)
    test_text_dataset = TxtDataSet(caption_files['test'], text_feat_path, opt)



    model = setup_model(opt)

    logger.info("Starting inference...")
    with torch.no_grad():
        score = eval_epoch(model, test_vid_dataset, test_text_dataset, opt)



if __name__ == '__main__':
    start_inference()