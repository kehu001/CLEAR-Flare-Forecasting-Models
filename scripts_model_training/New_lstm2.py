import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import pynvml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from New_SampleConstruction import get_samples
from utilities import TSS, HSS, POD, FAR, F1, ACC


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    

def normalize2(X, X_all):
    X_norm = np.copy(X)
    num_frames = X_all.shape[1]
    for i in range(num_frames):
        tmp = X_all[:, i, :]
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        X_norm[:, i, :] = (X_norm[:, i, :] - mean) / std
    return X_norm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size=30, truncation_size=1, num_layers=2, drop=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tr = truncation_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop)

        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        if self.tr > 1:
            out = out[:, -self.tr:, :].mean(dim=1)
        else:
            out = out[:, -1, :]
        return self.seq(out)


def compute_best_tss(y_true, y_scores, thresholds=np.linspace(0.30, 0.85, 10)):
    best_thresh = 0.5
    best_tss = -1.0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tss = TSS(y_pred, y_true)
        if tss > best_tss:
            best_tss = tss
            best_thresh = t
    return best_thresh, best_tss


# 2025/3/15 update: do not need downsample the negative ones since we do not inlude quiet samples anymore
# 2025/5/30 update: bootstrap the training data
class train_lstm:
    def __init__(self, device=DEVICE,n_booststrap=30):
        #self.criterion = nn.BCEWithLogitsLoss(torch.tensor([2.0]).to(device))
        self.device = device
        self.n_bootstrap = n_booststrap
        return
    
    def combine(self, pos_inputs, neg_inputs):
        if len(pos_inputs) == 0:
            targets = np.zeros(len(neg_inputs))
            return neg_inputs, targets
        if len(neg_inputs) == 0:
            targets = np.ones(len(pos_inputs))
            return pos_inputs, targets 
        else:
            inputs = np.concatenate([pos_inputs, neg_inputs])
            targets = np.concatenate([np.ones(len(pos_inputs)), np.zeros(len(neg_inputs))])
            # shuffle the data
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]
            return inputs, targets
    
    def sample_combine(self, pos_inputs, neg_inputs, pos_weight):
        if len(pos_inputs) == 0:
            # random choose 80% of the negative samples
            idx_rn = np.random.choice(len(neg_inputs), int(0.8*len(neg_inputs)), replace=False)
            neg_inputs = neg_inputs[idx_rn]
            targets = np.zeros(len(neg_inputs))
            # shuffle the data
            idx = np.arange(len(neg_inputs))
            np.random.shuffle(idx)
            neg_inputs = neg_inputs[idx]
            return neg_inputs, targets
        else:
            if len(neg_inputs) < pos_weight*len(pos_inputs):
                replace = True
            else:
                replace = False
            # random choose pos_weight times of the negative samples
            idx_rn = np.random.choice(len(neg_inputs), pos_weight*len(pos_inputs), replace=replace)
            neg_inputs = neg_inputs[idx_rn]
            inputs = np.concatenate([pos_inputs, neg_inputs])
            targets = np.concatenate([np.ones(len(pos_inputs)), np.zeros(len(neg_inputs))])
            # shuffle the data
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]
            return inputs, targets
        
    def train_lstm(self, model, trainloader, criterion, optimizer, device, n_epochs=20):
        model.train()
        train_loss = []
     
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, (data, label) in enumerate(trainloader):
                data, label = data.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.squeeze(), label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(trainloader)
            train_loss.append(avg_loss)
            print(f'Epoch {epoch+1}, avg loss: {avg_loss}')
            #scheduler.step(avg_loss)
        return train_loss
    
    def get_best_threshold(self, y_true, y_prob):
        thresholds = np.linspace(0, 1, 21)
        tss_scores = [TSS(y_prob >= t, y_true) for t in thresholds]
        return thresholds[np.argmax(tss_scores)], max(tss_scores)
    
    def train(self, inputs_profiles, labels, purpose, time1, time2, n_epoch=20, 
                    val=True, plot=True):
        # initialize monitering
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        process = psutil.Process()
        start_time = time.time()
        max_cpu_mem = 0
        max_gpu_mem = 0
        
        self.purpose = purpose
        pos_inputs, neg_inputs = get_samples(inputs_profiles, labels, self.purpose, time1, time2)
        inputs, targets = self.combine(pos_inputs, neg_inputs)
        self.full_inputs = inputs.copy()

        dim = self.full_inputs.shape[2]
        self.models = []
        self.thresholds = []
        self.tss_scores = []
        pos_weight = len(neg_inputs) / len(pos_inputs)
        self.criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(self.device))
        
        # normalize the inputs
        inputs = normalize2(inputs, self.full_inputs)

        if val:
            inputs, X_val, targets, y_val = train_test_split(
                inputs, targets, test_size=0.1, random_state=42, stratify=targets
            )
        
        cpu_mem = process.memory_info().rss / (1024 ** 2)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
        max_cpu_mem = cpu_mem
        max_gpu_mem = gpu_mem
        n_size = inputs.shape[0]
        for i in range(self.n_bootstrap):
            print(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            X_boot, y_boot = resample(inputs, targets, replace=True, n_samples= n_size, random_state=i)

            trainloader = DataLoader(MyDataset(X_boot, y_boot), batch_size=128, shuffle=True, drop_last=True)

            m = lstm(dim).to(self.device)
            optimizer = optim.Adam(m.parameters(), lr=0.001)

            train_loss = self.train_lstm(m, trainloader, self.criteria, optimizer, self.device, n_epochs=n_epoch)
            self.models.append(m)

            # uodate storage
            cpu_mem = process.memory_info().rss / (1024 ** 2)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
            max_cpu_mem = max(max_cpu_mem, cpu_mem)
            max_gpu_mem = max(max_gpu_mem, gpu_mem)

            # tss on validation set           
            if val:
                m.eval()
                val_scores = []
                with torch.no_grad():
                    val_out = m(torch.tensor(X_val, dtype=torch.float32).to(self.device))
                    val_out = torch.sigmoid(val_out).cpu().numpy().flatten()
                    best_threshold, best_tss = self.get_best_threshold(y_val, val_out)
                    #val_scores.append(val_out)
                    self.thresholds.append(best_threshold)
                    self.tss_scores.append(best_tss)

            if plot:
                plt.plot(train_loss)
        plt.show()
        if val:
            print(f"The 95% CI for threshold is {np.percentile(self.thresholds, 2.5)} - {np.percentile(self.thresholds, 97.5)}")
            print(f"The 95% CI for TSS is {np.percentile(self.tss_scores, 2.5)} - {np.percentile(self.tss_scores, 97.5)}")
        
        end_time = time.time()
        print(f"\n[Resource Summary for {self.n_bootstrap} bootstrapped LSTM models]")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        print(f"Max CPU memory used: {max_cpu_mem:.2f} MB")
        print(f"Max GPU memory used: {max_gpu_mem:.2f} MB")
        
        # check the convergence of thresholds and tss scores along the bootstrap iterations
        iterations = np.arange(1, self.n_bootstrap + 1)
        mean_thresholds = np.cumsum(self.thresholds) / iterations
        mean_tss_scores = np.cumsum(self.tss_scores) / iterations
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(mean_thresholds, label='Mean Threshold')
        plt.axhline(y=np.percentile(self.thresholds, 2.5), color='r', linestyle='--', label='95% CI Lower')
        plt.axhline(y=np.percentile(self.thresholds, 97.5), color='g', linestyle='--', label='95% CI Upper')
        plt.title('Thresholds Convergence on Validation Set with 95% CI')
        plt.xlabel('Bootstrap Iteration')
        plt.ylabel('Threshold')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(mean_tss_scores, label='Mean TSS')
        plt.axhline(y=np.percentile(self.tss_scores, 2.5), color='r', linestyle='--', label='95% CI Lower')
        plt.axhline(y=np.percentile(self.tss_scores, 97.5), color='g', linestyle='--', label='95% CI Upper')
        plt.title('TSS Convergence on Validation Set with 95% CI')
        plt.xlabel('Bootstrap Iteration')
        plt.ylabel('TSS')
        plt.legend()
        plt.tight_layout()
        plt.show()
    