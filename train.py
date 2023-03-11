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


parser = argparse.ArgumentParser(description='Training ProMOSmodel.')
parser.add_argument('--num_epochs', type=int, help='Number of epochs.', default=60)
parser.add_argument('--lamb_c', type=float, help='Weight of consistency loss lambda_c.', default=1.0)
parser.add_argument('--lamb_t', type=float, help='Weight of teacher model loss lambda_t.', default=0.0)
parser.add_argument('--log_valid', type=int, help='Logging valid score each log_valid epochs.', default=1)
parser.add_argument('--log_epoch', type=int, help='Logging training during a global run.', default=1)
parser.add_argument('--data_path', type=str, help='Path to data.', default='../testVCC2/')
parser.add_argument('--id_table', type=str, help='Path to ID of judges.', default='./id_table/')
parser.add_argument('--save_path', type=str, help='Path to save the model.', default='')
args = parser.parse_args()


def valid(model, 
          dataloader, 
          systems,
          steps,
          prefix,
          device,
          MSE_list,
          LCC_list,
          SRCC_list):
    model.eval()

    mos_predictions = []
    mos_targets = []
    mos_vars = []
    mos_predictions_sys = {system:[] for system in systems}
    mos_vars_sys = {system:[] for system in systems}
    true_sys_mean_scores = {system:[] for system in systems}

    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        wav, filename, _, mos, _ = batch
        sys_names = list(set([name.split("_")[0] for name in filename])) # system name, e.g. 'D03'
        wav = wav.to(device)
        wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, 257)

        with torch.no_grad():
            try:
                mos_prediction, mos_var = model(speech_spectrum = wav) # shape (batch, seq_len, 1)
                mos_prediction = mos_prediction.squeeze(-1) # shape (batch, seq_len)
                mos_var = mos_var.squeeze(-1)
                mos_prediction = torch.mean(mos_prediction, dim = -1) # torch.Size([1])
                mos_var = torch.mean(mos_var, dim = -1)

                mos_prediction = mos_prediction.cpu().detach().numpy()
                mos_var = mos_var.cpu().detach().numpy()
                mos_predictions.extend(mos_prediction.tolist())
                mos_targets.extend(mos.tolist())
                mos_vars.extend(mos_var.tolist())

                for j, sys_name in enumerate(sys_names):
                    mos_predictions_sys[sys_name].append(mos_prediction[j])
                    mos_vars_sys[sys_name].append(mos_var[j])
                    true_sys_mean_scores[sys_name].append(mos.tolist()[j])

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    mos_predictions = np.array(mos_predictions)
    mos_vars = np.array(mos_vars)
    mos_targets = np.array(mos_targets)

    mos_predictions_sys = np.array([np.mean(scores) for scores in mos_predictions_sys.values()])
    mos_vars_sys = np.array([np.mean(scores)/len(scores) for scores in mos_vars_sys.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])
    
    utt_MSE=np.mean((mos_targets-mos_predictions)**2)
    utt_LCC=np.corrcoef(mos_targets, mos_predictions)[0][1]
    utt_SRCC=scipy.stats.spearmanr(mos_targets, mos_predictions)[0]
    
    sys_MSE=np.mean((true_sys_mean_scores-mos_predictions_sys)**2)
    sys_LCC=np.corrcoef(true_sys_mean_scores, mos_predictions_sys)[0][1]
    sys_SRCC=scipy.stats.spearmanr(true_sys_mean_scores, mos_predictions_sys)[0]

    Likelihoods = []
    for i in range(len(mos_targets)):
        Likelihoods.append(stats.norm.pdf(mos_targets[i], mos_predictions[i], math.sqrt(mos_vars[i])))
    utt_GML=gmean(Likelihoods) # geometric mean of likelihood
    utt_AML=np.mean(Likelihoods) # arithemic mean of likelihood
    utt_MoV=np.mean(mos_vars) # mean of variance
    utt_VoV=np.var(mos_vars) # variance of variance

    Likelihoods_sys = []
    for i in range(len(true_sys_mean_scores)):
        Likelihoods_sys.append(stats.norm.pdf(true_sys_mean_scores[i], mos_predictions_sys[i], math.sqrt(mos_vars_sys[i])))
    sys_GML=gmean(Likelihoods_sys)
    sys_AML=np.mean(Likelihoods_sys)
    sys_MoV=np.mean(mos_vars_sys)
    sys_VoV=np.var(mos_vars_sys)
    
    MSE_list.append(utt_MSE)
    LCC_list.append(utt_LCC) 
    SRCC_list.append(utt_SRCC)
    print(
        f"\n[{prefix}][{steps}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]"
    )
    print(f"[{prefix}][{steps}][UTT][ GML = {utt_GML:.6f} | AML = {utt_AML:.6f} | MoV = {utt_MoV:.6f} | VoV = {utt_VoV:.6f} ] [SYS][ GML = {sys_GML:.6f} | AML = {sys_AML:.6f} | MoV = {sys_MoV:.6f} | VoV = {sys_VoV:.6f} ]")

    model.train()
    return MSE_list, LCC_list, SRCC_list, sys_SRCC


def train(num_epochs,
          lamb_c,
          lamb_t,
          log_valid,
          log_epoch,
          train_set,
          valid_set,
          test_set,
          train_loader,
          valid_loader,
          test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProMOSNet().to(device)
    momentum_model = ProMOSNet().to(device) 
    for param in momentum_model.parameters(): 
        param.detach_()
    momentum_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    optimizer_momentum = WeightExponentialMovingAverage(model, momentum_model) 
    optimizer.zero_grad()
    criterion1 = F.gaussian_nll_loss
    criterion2 = F.mse_loss

    backward_steps = 0
    all_loss = []

    best_LCC = -1
    best_LCC_teacher = -1
    best_sys_SRCC = -1
    
    MSE_list = []
    LCC_list = []
    SRCC_list = []
    train_loss = []
    MSE_teacher, LCC_teacher, SRCC_teacher = [], [], []
    MSE_student, LCC_student, SRCC_student = [], [], []
    
    model.train()
    epoch = 0
    while epoch <= num_epochs:
        if epoch == 5:
            optimizer_momentum.set_alpha(alpha = 0.999) 
        
        for i, batch in enumerate(tqdm(train_loader, ncols=0, desc="Train", unit=" step")):
            try:
                wavs, filename, _, mos, _ = batch
                wavs = wavs.to(device)
                wavs = wavs.unsqueeze(1) # shape (batch, 1, seq_len, 257[dim feature])
                mos = mos.to(device) # shape (batch)

                # Stochastic Gradient Noise (SGN)
                label_noise = torch.randn(mos.size(), device=device) # standard normal distribution
                mos += 0.1*label_noise

                # Forward
                mos_mean, mos_var = model(speech_spectrum=wavs) # (batch, seq_len, 1), (batch, seq_len, 1) 
                mos_mean_mom, mos_var_mom = momentum_model(speech_spectrum=wavs)
                mos_mean = mos_mean.squeeze() # (batch, seq_len)
                mos_var = mos_var.squeeze() # (batch, seq_len)
                mos_mean_mom = mos_mean_mom.squeeze() 
                mos_var_mom = mos_var_mom.squeeze() 
                seq_len = mos_mean.shape[1]
                bsz = mos_mean.shape[0]

                mos = mos.unsqueeze(1).repeat(1, seq_len) # (batch, seq_len) by repeat seq_len times

                # Loss
                loss = criterion1(mos_mean, mos, mos_var) # torch.Size([])
                cost_mean = criterion2(mos_mean, mos_mean_mom)
                cost_var = criterion2(mos_var, mos_var_mom)
                loss_teacher = criterion1(mos_mean_mom, mos, mos_var_mom)
                loss = loss + lamb_c*(cost_mean+cost_var) +lamb_t*loss_teacher
                
                # Backwards
                loss.backward()

                all_loss.append(loss.item())
                del loss

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                optimizer_momentum.step() 
                optimizer.zero_grad()
    
            except Exception as e:
                print(e)

        if epoch % log_epoch == 0:
            average_loss = torch.FloatTensor(all_loss).mean().item()
            train_loss.append(average_loss)
            print(f"Average loss={average_loss}")
            all_loss = []

        if epoch % log_valid == 0:
            MSE_teacher, LCC_teacher, SRCC_teacher, sys_SRCC_teacher = valid(momentum_model,
                                                                     valid_loader,
                                                                     valid_set.systems,
                                                                     epoch,
                                                                     'Valid(teacher)',
                                                                     device,
                                                                     MSE_teacher,
                                                                     LCC_teacher,
                                                                     SRCC_teacher)
            # MSE_student, LCC_student, SRCC_student, sys_SRCC_student = valid(model,
            #                                                         valid_loader,
            #                                                         valid_set.systems,
            #                                                         epoch,
            #                                                         'Valid(student)',
            #                                                         device,
            #                                                         MSE_student,
            #                                                         LCC_student,
            #                                                         SRCC_student)

            if LCC_teacher[-1] > best_LCC_teacher:
                best_LCC_teacher = LCC_teacher[-1]
                best_model = copy.deepcopy(momentum_model)

        epoch += 1

    print('Best model performance test:')
    _, _, _, _ = valid(best_model, test_loader, test_set.systems, epoch, 'Test(best)', device, MSE_teacher, LCC_teacher, SRCC_teacher)
    _, _, _, _ = valid(momentum_model, test_loader, test_set.systems, epoch, 'Test(last)', device, MSE_teacher, LCC_teacher, SRCC_teacher)
    return best_model, momentum_model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher

def main():
    data_path = args.data_path
    id_table = args.id_table
    train_set = get_dataset(data_path,
                            "training_data.csv",
                            vcc18=True,
                            idtable=os.path.join(id_table, 'idtable.pkl'))
    valid_set = get_dataset(data_path,
                            "valid_data.csv",
                            vcc18=True,
                            valid=True,
                            idtable=os.path.join(id_table, 'idtable.pkl'))
    test_set = get_dataset(data_path,
                           "testing_data.csv",
                           vcc18=True,
                           valid=True,
                           idtable=os.path.join(id_table, 'idtable.pkl'))
    train_loader = get_dataloader(train_set, batch_size=64, num_workers=1)
    valid_loader = get_dataloader(valid_set, batch_size=1, num_workers=1)
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=1)
    
    best_model, last_model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher = train(
        args.num_epochs, args.lamb_c, args.lamb_t, args.log_valid, args.log_epoch, train_set,
        valid_set, test_set, train_loader, valid_loader, test_loader)

    torch.save(best_model, args.save_path+'best.pt')
    torch.save(last_model, args.save_path+'last.pt')

main()
