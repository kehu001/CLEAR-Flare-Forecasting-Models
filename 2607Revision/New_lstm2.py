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
# 2026/7/27 update: add the pos_weight option to control the negative-to-positive ratio in training
class train_lstm:
    def __init__(self, device=DEVICE, n_bootstrap=30, pos_weight=None):
        self.pos_weight = pos_weight
        #self.criterion = nn.BCEWithLogitsLoss(torch.tensor([2.0]).to(device))
        self.device = device
        self.n_bootstrap = n_bootstrap
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
    
    '''def sample_combine(self, pos_inputs, neg_inputs, weight):
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
            if len(neg_inputs) < weight*len(pos_inputs):
                replace = True
            else:
                replace = False
            # random choose weight times of the negative samples
            idx_rn = np.random.choice(len(neg_inputs), int(weight*len(pos_inputs)), replace=replace)
            neg_inputs = neg_inputs[idx_rn]
            inputs = np.concatenate([pos_inputs, neg_inputs])
            targets = np.concatenate([np.ones(len(pos_inputs)), np.zeros(len(neg_inputs))])
            # shuffle the data
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]
            return inputs, targets'''
        
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
        """
    Train bootstrap LSTM models.

    Classification/training modes
    -----------------------------
    self.pos_weight is None:
        Use all positive and negative training samples.
        For each bootstrap, set

            pos_weight = n_negative / n_positive

        based on the full training data.

    self.pos_weight is a positive number, e.g. 2:
        For each bootstrap, use all positive samples and randomly
        undersample negatives so that

            n_negative = self.pos_weight * n_positive.

        The same value is used as BCEWithLogitsLoss(pos_weight=...).

        For example, self.pos_weight = 2 gives a 2:1 negative-to-positive
        training ratio and pos_weight=2.
    """
        # initialize monitering
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        process = psutil.Process()
        start_time = time.time()
        max_cpu_mem = 0
        max_gpu_mem = 0
        
        self.purpose = purpose
        pos_inputs, neg_inputs = get_samples(inputs_profiles, labels, self.purpose, time1, time2)
        print(f"get samples")
        if len(pos_inputs) == 0:
            raise ValueError("No positive samples were found.")
        if len(neg_inputs) == 0:
            raise ValueError("No negative samples were found.")
        inputs, targets = self.combine(pos_inputs, neg_inputs)
        
        #self.full_inputs = inputs.copy()

        dim = inputs.shape[2]
        self.models = []
        self.thresholds = []
        self.tss_scores = []

        # split the data into training and validation sets
        if val:
            train_inputs, X_val_raw, train_targets, y_val = train_test_split(
            inputs,
            targets,
            test_size=0.1,
            random_state=42,
            stratify=targets
        )
        else:
            train_inputs = inputs
            train_targets = targets
            X_val_raw = None
            y_val = None
        
        # normalize the inputs
        train_reference = train_inputs.copy()

        train_inputs = normalize2(
            train_inputs,
            train_reference
        )

        if val:
            X_val = normalize2(
                X_val_raw,
                train_reference
            )

        # Separate positive and negative training samples
        positive_mask = train_targets == 1
        negative_mask = train_targets == 0

        X_pos = train_inputs[positive_mask]
        y_pos = train_targets[positive_mask]

        X_neg = train_inputs[negative_mask]
        y_neg = train_targets[negative_mask]

        n_pos = len(X_pos)
        n_neg = len(X_neg)

        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                "The training split must contain both positive and negative samples."
            )

        # ---------------------------------------------------------
        # Determine the training mode
        # ---------------------------------------------------------
        if self.pos_weight is None:
            # Use all available negatives.
            negative_sampling_ratio = n_neg / n_pos
            loss_pos_weight = n_neg / n_pos

            print(
                "Using all training samples: "
                f"{n_pos} positives and {n_neg} negatives."
            )
            print(f"BCE positive-class weight: {loss_pos_weight:.4f}")

        else:
            if not isinstance(self.pos_weight, (int, float)):
                raise TypeError(
                    "self.pos_weight must be None or a positive number."
                )

            if self.pos_weight <= 0:
                raise ValueError(
                    "self.pos_weight must be greater than zero."
                )

            negative_sampling_ratio = float(self.pos_weight)
            loss_pos_weight = float(self.pos_weight)

            requested_n_neg = int(round(negative_sampling_ratio * n_pos))

            print(
                "Using negative undersampling with "
                f"{negative_sampling_ratio:g} negatives per positive."
            )
            print(
                f"Each bootstrap will contain approximately {n_pos} positives "
                f"and {min(requested_n_neg, n_neg)} negatives."
            )
            print(f"BCE positive-class weight: {loss_pos_weight:g}")

        
        cpu_mem = process.memory_info().rss / (1024 ** 2)
        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)
        max_cpu_mem = cpu_mem
        max_gpu_mem = gpu_mem
        
        # booststrap training
        for i in range(self.n_bootstrap):
            print(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            # Bootstrap positive samples with replacement
            X_pos_boot, y_pos_boot = resample(
                X_pos,
                y_pos,
                replace=True,
                n_samples=n_pos,
                random_state=i
            )

            if self.pos_weight is None:
                # Bootstrap all negative samples with replacement
                X_neg_boot, y_neg_boot = resample(
                    X_neg,
                    y_neg,
                    replace=True,
                    n_samples=n_neg,
                    random_state=10_000 + i
                )
            else:
                requested_n_neg = int(
                round(negative_sampling_ratio * len(X_pos_boot))
            )

                # Because negatives are abundant, sample without replacement.
                # If the requested number exceeds the available negatives,
                # use all available negatives.
                n_neg_boot = min(requested_n_neg, n_neg)

                X_neg_boot, y_neg_boot = resample(
                    X_neg,
                    y_neg,
                    replace=False,
                    n_samples=n_neg_boot,
                    random_state=10_000 + i
                )

            # Combine positive and negative bootstrap samples
            X_boot = np.concatenate(
                [X_pos_boot, X_neg_boot],
                axis=0
            )

            y_boot = np.concatenate(
                [y_pos_boot, y_neg_boot],
                axis=0
            )

            # Shuffle the combined bootstrap sample
            rng = np.random.default_rng(seed=20_000 + i)
            shuffle_indices = rng.permutation(len(y_boot))

            X_boot = X_boot[shuffle_indices]
            y_boot = y_boot[shuffle_indices]

            # Use the actual sampled ratio for numerical consistency
            n_pos_boot = np.sum(y_boot == 1)
            n_neg_boot = np.sum(y_boot == 0)
            bootstrap_pos_weight = n_neg_boot / n_pos_boot

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(bootstrap_pos_weight, dtype=torch.float32, device=self.device
                )
            )

            trainloader = DataLoader(
                MyDataset(X_boot, y_boot), batch_size=128, shuffle=True, drop_last=True
            )

            # Initialize model and optimizer
            m = lstm(dim).to(self.device)
            optimizer = optim.Adam(m.parameters(), lr=0.001)

            train_loss = self.train_lstm(m, trainloader, criterion, optimizer, self.device, n_epochs=n_epoch)
            self.models.append(m)

            # update storage
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
            print(f"The chosen threshold: {np.mean(self.thresholds)}")
            print(f"The 95% CI for TSS is {np.percentile(self.tss_scores, 2.5)} - {np.percentile(self.tss_scores, 97.5)}")
        
        end_time = time.time()
        print(f"\n[Resource Summary for {self.n_bootstrap} bootstrapped LSTM models]")
        print(f"Total training time: {end_time - start_time:.2f} seconds")
        print(f"Max CPU memory used: {max_cpu_mem:.2f} MB")
        print(f"Max GPU memory used: {max_gpu_mem:.2f} MB")

        pynvml.nvmlShutdown()
        
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
    