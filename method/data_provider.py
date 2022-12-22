import json
import torch
import torch.utils.data as data
import numpy as np
import re
import h5py


def getVideoId(cap_id):
    vid_id = cap_id.split('#')[0]
    return vid_id

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    return string.strip().lower().split()

def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def average_to_fixed_length(visual_input, map_size):
    visual_input = torch.from_numpy(visual_input)
    num_sample_clips = map_size
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips + 1, 1.0) / num_sample_clips * num_clips

    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips - 1))

    new_visual_input = []

    for i in range(num_sample_clips):

        s_idx, e_idx = idxs[i].item(), idxs[i + 1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0).numpy()


    return new_visual_input


def l2_normalize_np_array(np_array, eps=1e-5):
    """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
    return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)



def collate_train(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    frame_video_features, captions, idxs, cap_ids, video_ids = zip(*data)

    # videos
    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    #captions
    feat_dim = captions[0][0].shape[-1]

    merge_captions = []
    all_lengths = []
    labels = []

    for index, caps in enumerate(captions):
        labels.extend(index for i in range(len(caps)))
        all_lengths.extend(len(cap) for cap in caps)
        merge_captions.extend(cap for cap in caps)

    target = torch.zeros(len(all_lengths), max(all_lengths), feat_dim)
    words_mask = torch.zeros(len(all_lengths), max(all_lengths))

    for index, cap in enumerate(merge_captions):
        end = all_lengths[index]
        target[index, :end, :] = cap[:end, :]
        words_mask[index, :end] = 1.0



    return dict(frame_video_features=frame_videos,
                videos_mask=videos_mask,
                text_feat=target,
                text_mask=words_mask,
                text_labels=labels
                )


def collate_frame_val(data):
    frame_video_features, idxs, video_ids = zip(*data)

    video_lengths = [len(frame) for frame in frame_video_features]
    frame_vec_len = len(frame_video_features[0][0])
    frame_videos = torch.zeros(len(frame_video_features), max(video_lengths), frame_vec_len)
    videos_mask = torch.zeros(len(frame_video_features), max(video_lengths))
    for i, frames in enumerate(frame_video_features):
        end = video_lengths[i]
        frame_videos[i, :end, :] = frames[:end, :]
        videos_mask[i, :end] = 1.0

    return frame_videos, videos_mask, idxs, video_ids


def collate_text_val(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths), captions[0].shape[-1])
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    return target, words_mask, idxs, cap_ids




class Dataset4Training(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, video_feat_path, text_feat_path, opt, video2frames=None):
        # Captions
        self.captions = {}
        self.cap_ids = []
        self.video_ids = []
        self.vid_caps = {}
        # self.video2frames = video2frames

        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
                if video_id not in self.video_ids:
                    self.video_ids.append(video_id)
                if video_id in self.vid_caps:
                    self.vid_caps[video_id].append(cap_id)
                else:
                    self.vid_caps[video_id] = []
                    self.vid_caps[video_id].append(cap_id)
        # self.visual_feat = visual_feat
        self.text_feat_path = text_feat_path
        # new vid feat
        self.visual_feat = h5py.File(video_feat_path, 'r')

        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l
        self.max_desc_len = opt.max_desc_l

        self.open_file = False
        self.length = len(self.vid_caps)



    def __getitem__(self, index):

        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True
        video_id = self.video_ids[index]
        cap_ids = self.vid_caps[video_id]

        frame_vecs = self.visual_feat[video_id][...]

        frame_video_feature = average_to_fixed_length(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        # text
        cap_tensors = []
        for cap_id in cap_ids:

            cap_feat = self.text_feat[cap_id][...]

            cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]
            cap_tensors.append(cap_tensor)

        return frame_video_feature, cap_tensors, index, cap_ids, video_id

    def __len__(self):
        return self.length

class VisDataSet(data.Dataset):

    def __init__(self, video_feat_path, opt, video_ids=None):
        # self.visual_feat = visual_feat
        # self.video2frames = video2frames
        self.visual_feat = h5py.File(video_feat_path, 'r')
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = None
        self.length = len(self.video_ids)
        self.map_size = opt.map_size
        self.max_ctx_len = opt.max_ctx_l

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # frame_list = self.video2frames[video_id]
        # frame_vecs = []
        # for frame_id in frame_list:
        #     frame_vecs.append(self.visual_feat.read_one(frame_id))
        # new vid feat:
        frame_vecs = self.visual_feat[video_id][...]

        frame_video_feature = average_to_fixed_length(frame_vecs, self.max_ctx_len)
        frame_video_feature = l2_normalize_np_array(frame_video_feature)
        frame_video_feature = torch.from_numpy(frame_video_feature)

        return frame_video_feature, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet(data.Dataset):
    """
    Load captions
    """

    def __init__(self, cap_file, text_feat_path, opt):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.text_feat_path = text_feat_path
        self.max_desc_len = opt.max_desc_l
        self.open_file = False
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        if self.open_file:
            self.open_file = True
        else:
            self.text_feat = h5py.File(self.text_feat_path, 'r')

            self.open_file = True


        cap_feat = self.text_feat[cap_id][...]

        cap_tensor = torch.from_numpy(l2_normalize_np_array(cap_feat))[:self.max_desc_len]

        return cap_tensor, index, cap_id

    def __len__(self):
        return self.length




if __name__ == '__main__':
    pass


