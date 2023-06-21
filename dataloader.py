import os

import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, test_mode=False, sampling='random', transform=None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        self.num_segments = 32
        self.sampling = sampling
        self.test_mode = test_mode

        self.normal_root_path_train = ".../colon_i3d_feature_train_normal"
        self.abnormal_root_path_train = ".../colon_i3d_feature_train_abnormal"

        self.normal_root_path_test = ".../colon_i3d_feature_test_normal"
        self.abnormal_root_path_test = ".../colon_i3d_feature_test_abnormal"

        self.tranform = transform
        self.list = self._parse_list()
        self.num_frame = 0
        self.labels = None

    def _parse_list(self):
        normal_file_list = sorted(os.listdir(self.normal_root_path_train))
        abnormal_file_list = sorted(os.listdir(self.abnormal_root_path_train))

        normal_file_list_test = sorted(os.listdir(self.normal_root_path_test))
        abnormal_file_list_test = sorted(os.listdir(self.abnormal_root_path_test))
        if self.test_mode is False:
            if self.is_normal:
                l = [self.normal_root_path_train + '/' + s for s in normal_file_list]
            else:
                l = [self.abnormal_root_path_train + '/' + s for s in abnormal_file_list]
        else:
            l = [self.normal_root_path_test + '/' + s for s in normal_file_list_test] + [
                self.abnormal_root_path_test + '/' + s for s in abnormal_file_list_test]
        return l

    def __getitem__(self, index):
        label = self.get_label()
        file_name = self.list[index].strip('\n')
        features = np.load(file_name, allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features, file_name

        features = features.transpose(1, 0, 2)
        features = process_feat(features.squeeze(0), 32)  # divide a video into 32 segments
        features = np.array(features, dtype=np.float32)
        features = np.expand_dims(features, 1)

        return features, label

    def get_data(self, index):
        feature = np.load(self.list[index].strip('\n'), allow_pickle=True)
        vid_num_seg = feature.shape[0]
        if self.sampling == 'random':
            sample_idx = self.random_perturb(feature.shape[0])
        elif self.sampling == 'uniform':
            sample_idx = self.uniform_sampling(feature.shape[0])
        else:
            raise AssertionError('Not supported sampling !')

        feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_num_seg, sample_idx

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


def get_dataloader(args):
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)
    return train_nloader, train_aloader, test_loader
