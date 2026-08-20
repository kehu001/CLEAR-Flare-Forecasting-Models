import numpy as np
import pandas as pd
#import h5py
from astropy.time import Time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

params = ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBZ', 'MEANGBH', 'MEANJZD',
                'TOTUSJZ', 'MEANALP', 'MEANJZH', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP', 'MEANPOT',
                'TOTPOT', 'MEANSHR', 'SHRGT45', 'SIZE', 'SIZE_ACR', 'NACR', 'NPIX']


def rescale(label):
    c = label[0]
    intensity = float(label[1:])
    if c == 'B':
        return np.log10(intensity * 10 ** -7 / 0.7)
    elif c == 'C':
        return np.log10(intensity * 10 ** -6 / 0.7)
    elif c == 'M':
        return np.log10(intensity * 10 ** -5 / 0.7)
    elif c == 'A':
        return np.log10(intensity * 10 ** -8 / 0.7)
    else:
        return np.log10(intensity * 10 ** -4 / 0.7)

def rescale_back(label):
    c = label[0]
    intensity = float(label[1:])
    if c == 'B':
        return np.log10(intensity * 10 ** -7 * 0.7)
    elif c == 'C':
        return np.log10(intensity * 10 ** -6 * 0.7)
    elif c == 'M':
        return np.log10(intensity * 10 ** -5 * 0.7)
    elif c == 'A':
        return np.log10(intensity * 10 ** -8 * 0.7)
    else:
        return np.log10(intensity * 10 ** -4 * 0.7)

def TAI2UTC(tai_ts):
    # input: a single timestamp in TAI scale, with the datetime type
    # output: a single timestamp in UTC scale, with the datetime type
        ts = Time(tai_ts, format='datetime', scale='tai')
        ts = ts.utc
        return ts.value
    

def get_log10_intensity(label):
    # Turn class to continuous log intensity.
    # input: flare intensity in the form of CLASS+level, e.g. 'M1.8'
    # output: log10(flare intensity)
    c = label[0]
    intensity = float(label[1:])
    if c == 'B':
        return np.log10(intensity * 10 ** -7)
    elif c == 'C':
        return np.log10(intensity * 10 ** -6)
    elif c == 'M':
        return np.log10(intensity * 10 ** -5)
    elif c == 'A':
        return np.log10(intensity * 10 ** -8)
    else:
        return np.log10(intensity * 10 ** -4)

def get_mag(log_intensity):
    if log_intensity < -7.0:
        return 'A'
    elif log_intensity < -6.0:
        return 'B'
    elif log_intensity < -5.0:
        return 'C'
    elif log_intensity < -4.0:
        return 'M'
    else:
        return 'X'


def normalize2(X, X_all):
    X_norm = np.copy(X)
    num_frames = X_all.shape[1]
    for i in range(num_frames):
        tmp = X_all[:, i, :]
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        X_norm[:, i, :] = (X_norm[:, i, :] - mean) / std
    return X_norm

def normalize_lr(X, X_all):
    mean = np.mean(X_all, axis=0)
    std = np.std(X_all, axis=0)
    return (X - mean)/std

def getLabel(output):
    output = torch.sigmoid(output)
    return (output > 0.5).int() 

def TSS(pred, target):
    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))
    return (TP) / (TP + FN ) - (FP) / (FP + TN ) 

def HSS(pred, target):
    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))
    return 2 * (TP * TN - FP * FN ) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

def POD(pred, target):
    TP = np.sum((pred == 1) & (target == 1))
    FN = np.sum((pred == 0) & (target == 1))
    return TP / (TP + FN)

def FAR(pred, target):
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    return FP / (TN + FP)

def F1(pred, target):
    TP = np.sum((pred == 1) & (target == 1))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))
    return 2 * TP / (2 * TP + FP + FN)

def ACC(pred, target):
    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    return (TP + TN) / len(pred)

def combine(pos_inputs, neg_inputs):
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