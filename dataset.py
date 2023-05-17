import os
from tqdm import tqdm
import librosa
import numpy as np
from collections import defaultdict
import scipy
import h5py
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class VCC16Dataset(Dataset):
    def __init__(self, data_path, labels = None):
        self.data_path = data_path
        self.wav_name = [f for f in os.listdir(self.data_path) if 'wav' in f]
        self.length = len(self.wav_name)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wav_path = os.path.join(
            self.data_path,
            self.wav_name[idx],
        )
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = np.abs(librosa.stft(wav, n_fft = 512)).T
        return wav, self.wav_name[idx]

    def collate_fn(self, batch):
        wavs, wav_names = zip(*batch)
        wavs = list(wavs)
        wav_names = list(wav_names)
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        return output_wavs, wav_names

class VCC18Dataset(Dataset):
    def __init__(self, wav_file, score_csv, idtable = '', valid = False):
        self.wavs = wav_file # list of 257-dim features
        self.scores = score_csv # dataframe
        
        self.systems = list(set([name.split("_")[0] for name in self.scores["WAV_PATH"]])) # 'N14', 'N10', etc.
        
        if os.path.isfile(idtable):
            self.idtable = torch.load(idtable) # a dictionary from judge code to id number, 0 to 269
            for i, judge_i in enumerate(score_csv['JUDGE']):
                self.scores['JUDGE'][i] = self.idtable[judge_i]
        elif not valid:
            self.gen_idtable(idtable) # generate for training data, but actually the same as the one read
            
    def __getitem__(self, idx):
        if type(self.wavs[idx]) == int:
            wav = self.wavs[idx - self.wavs[idx]]
        else:
            wav = self.wavs[idx]
        return wav, self.scores['WAV_PATH'][idx], self.scores['MEAN'][idx], self.scores['MOS'][idx], self.scores['JUDGE'][idx]
    
    def __len__(self):
        return len(self.wavs)

    def gen_idtable(self, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 0
        for i, judge_i in enumerate(self.scores['JUDGE']):
            if judge_i not in self.idtable.keys():
                self.idtable[judge_i] = count
                count += 1
                self.scores['JUDGE'][i] = self.idtable[judge_i]
            else:
                self.scores['JUDGE'][i] = self.idtable[judge_i]
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, samples):
        # wavs may be list of wave or spectrogram, which has shape (time, feature) or (time,)
        wavs, filenames, means, scores, judge_ids = zip(*samples)
        max_len = max(wavs, key = lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for i, wav in enumerate(wavs):
            wav_len = wav.shape[0]
            dup_times = max_len//wav_len
            remain = max_len - wav_len*dup_times
            to_dup = [wav for t in range(dup_times)]
            to_dup.append(wav[:remain, :])
            output_wavs.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        output_wavs = torch.stack(output_wavs, dim = 0)
        means = torch.FloatTensor(means)
        scores = torch.FloatTensor(scores)
        judge_ids = torch.LongTensor(judge_ids)
        return output_wavs, filenames, judge_ids, means, scores

class BVCCDataset(Dataset):
    def __init__(self, metadata, data_path, idtable='', split="train", valid=False):
        self.data_path = data_path
        self.split = split

        # cache features
        self.features = {}

        # get judge id table and number of judges
        if os.path.isfile(idtable):
            self.idtable = torch.load(idtable)
        elif not valid:
            self.gen_idtable(metadata, idtable)
        self.num_judges = len(self.idtable)

        self.metadata = []
        if self.split == "train":
            for wav_name, judge_name, avg_score, score in metadata:
                self.metadata.append([wav_name, avg_score, score, self.idtable[judge_name]])
        else:
            for item in metadata:
                self.metadata.append(item)

            # build system list
            self.systems = list(set([item[0] for item in metadata]))
            
    def __getitem__(self, idx):
        if self.split == "train":
            wav_name, avg_score, score, judge_id = self.metadata[idx]
        else:
            sys_name, wav_name, avg_score = self.metadata[idx]

        # cache features
        if wav_name in self.features:
            mag_sgram = self.features[wav_name]
        else:
            h5_path = os.path.join(self.data_path, "bin", wav_name + ".h5")
            if os.path.isfile(h5_path):
                data_file = h5py.File(h5_path, 'r')
                mag_sgram = np.array(data_file['mag_sgram'][:])
                timestep = mag_sgram.shape[0]
                mag_sgram = np.reshape(mag_sgram,(timestep, 257))
            else:
                wav, _ = librosa.load(os.path.join(self.data_path, "wav", wav_name), sr = 16000)
                mag_sgram = np.abs(librosa.stft(wav, n_fft = 512, hop_length=256, win_length=512, window=scipy.signal.hamming)).astype(np.float32).T
            self.features[wav_name] = mag_sgram

        if self.split == "train":
            return mag_sgram, avg_score, score, judge_id
        else:
            return mag_sgram, avg_score, sys_name, wav_name

    def __len__(self):
        return len(self.metadata)

    def gen_idtable(self, metadata, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 0
        for _, judge_name, _, _ in metadata:
            if judge_name not in self.idtable:
                self.idtable[judge_name] = count
                count += 1
        torch.save(self.idtable, idtable_path)

    def collate_fn(self, batch):
        sorted_batch = sorted(batch, key=lambda x: -x[0].shape[0])
        bs = len(sorted_batch) # batch_size
        avg_scores = torch.FloatTensor([sorted_batch[i][1] for i in range(bs)])
        mag_sgrams = [torch.from_numpy(sorted_batch[i][0]) for i in range(bs)]
        mag_sgrams_lengths = torch.from_numpy(np.array([mag_sgram.size(0) for mag_sgram in mag_sgrams]))
        
        # repetitive padding
        max_len = mag_sgrams_lengths[0]
        mag_sgrams_padded = []
        for mag_sgram in mag_sgrams:
            this_len = mag_sgram.shape[0]
            dup_times = max_len // this_len
            remain = max_len - this_len * dup_times
            to_dup = [mag_sgram for t in range(dup_times)]
            to_dup.append(mag_sgram[:remain, :])
            mag_sgrams_padded.append(torch.Tensor(np.concatenate(to_dup, axis = 0)))
        mag_sgrams_padded = torch.stack(mag_sgrams_padded, dim = 0)

        if not self.split == "train":
            sys_names = [sorted_batch[i][2] for i in range(bs)]
            return mag_sgrams_padded, avg_scores, sys_names
        else:
            scores = torch.FloatTensor([sorted_batch[i][2] for i in range(bs)])
            judge_ids = torch.LongTensor([sorted_batch[i][3] for i in range(bs)])
            return mag_sgrams_padded, mag_sgrams_lengths, judge_ids, avg_scores, scores


def get_dataset(data_path, split, vcc18 = False, bvcc = False, idtable = '', valid = False):
    if vcc18:
        dataframe = pd.read_csv(os.path.join(data_path, f'{split}'), index_col=False)
        wavs = []
        filename = ''
        last = 0
        print("Loading all wav files.")
        for i in tqdm(range(len(dataframe))):
            if dataframe['WAV_PATH'][i] != filename:
                wav, _ = librosa.load(os.path.join(data_path, dataframe['WAV_PATH'][i]), sr = 16000)
                wav = np.abs(librosa.stft(wav, n_fft = 512)).T
                wavs.append(wav)
                filename = dataframe['WAV_PATH'][i]
                last = 0
            else:
                last += 1
                wavs.append(last)

        return VCC18Dataset(wav_file=wavs, score_csv = dataframe, idtable = idtable, valid = valid)
    elif bvcc:
        names = {"train":"TRAINSET", "valid":"DEVSET", "test":"TESTSET"}
    
        metadata = defaultdict(dict)
        metadata_with_avg = list()

        # read metadata
        with open(os.path.join(data_path, "sets", names[split]), "r") as f:
            lines = f.read().splitlines()
           
            # line has format <system, wav_name, score, _, judge_name>
            for line in lines:
                if line:
                    parts = line.split(",")
                    sys_name = parts[0]
                    wav_name = parts[1]
                    score = int(parts[2])
                    judge_name = parts[4]
                    metadata[sys_name + "|" + wav_name][judge_name] = score
        
        # calculate average score
        for _id, v in metadata.items():
            sys_name, wav_name = _id.split("|")
            avg_score = np.mean(np.array(list(v.values())))
            if split == "train":
                for judge_name, score in v.items():
                    metadata_with_avg.append([wav_name, judge_name, avg_score, score])
            else:
                # in testing mode, additionally return system name and only average score
                metadata_with_avg.append([sys_name, wav_name, avg_score])

        return BVCCDataset(metadata_with_avg, data_path, idtable, split, valid = valid)
    return VCC16Dataset(data_path)

def get_dataloader(dataset, batch_size, num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
