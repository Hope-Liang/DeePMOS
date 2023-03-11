import argparse
import copy
import math
import os
import random
import typing

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import gmean

from dataset import get_dataloader, get_dataset
from EMA import WeightExponentialMovingAverage
from model import ProMOSNet


parser = argparse.ArgumentParser(description='Testing ProMOSmodel.')
parser.add_argument('--data_path', type=str, help='Path to data.', default='../testVCC2/')
parser.add_argument('--id_table', type=str, help='Path to ID of judges.', default='./id_table/')
parser.add_argument('--model_path', type=str, help='Path to load the model.', default='')
args = parser.parse_args()

def main():
    data_path = args.data_path
    model_path = args.model_path
    id_table = args.id_table
    test_set = get_dataset(data_path,
                           "testing_data.csv",
                           vcc18=True,
                           valid=True,
                           idtable=os.path.join(id_table, 'idtable.pkl'))
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path,map_location=device)

    mos_means = []
    mos_vars = []
    mos_s = []
    for i, batch in enumerate(tqdm(test_loader, ncols=0, unit=" step")):
        wav, filename, _, mos, _ = batch
        wav = wav.to(device)
        wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, 257)
        mos = mos.to(device)

        with torch.no_grad():
            mos_mean, mos_var = model(speech_spectrum = wav) # shape (batch, seq_len, 1)
            mos_mean = mos_mean.squeeze(-1) # shape (batch, seq_len)
            mos_var = mos_var.squeeze(-1) # shape (batch, seq_len)
            mos_mean = torch.mean(mos_mean, dim = -1) # torch.Size([1])
            mos_var = torch.mean(mos_var, dim = -1) # torch.Size([1])
            
            mos_s.extend(mos.cpu().detach().numpy().tolist())
            mos_means.extend(mos_mean.cpu().detach().numpy().tolist())
            mos_vars.extend(mos_var.cpu().detach().numpy().tolist())

    LH = []
    CI95 = 0
    for i in range(len(mos_s)):
        LH.append(stats.norm.pdf(mos_s[i], mos_means[i], math.sqrt(mos_vars[i])))
        lb = mos_means[i]-1.96*math.sqrt(mos_vars[i])
        ub = mos_means[i]+1.96*math.sqrt(mos_vars[i])
        if mos_s[i] > lb and mos_s[i] < ub:
            CI95 += 1

    q1,q2,q3 = np.percentile(LH, [25, 50, 75])
    print("First quantile {}, second quantile {}, third quantile {}".format(q1,q2,q3))

    CI95 = CI95/4000
    print("CI95 ", CI95)


main()
