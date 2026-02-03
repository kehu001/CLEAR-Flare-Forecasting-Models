import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import psutil
import pynvml

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


from New_SampleConstruction import get_samples
from utilities import TSS, HSS, POD, FAR, F1, ACC


class train_logistic:
    def __init__(self, n_booststrap=30):
        self.n_bootstrap = n_booststrap
        self.models = []
        self.purpose = None

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
    
    def get_best_threshold(self, y_true, y_prob):
        thresholds = np.linspace(0, 1, 21)
        tss_scores = [TSS(y_prob >= t, y_true) for t in thresholds]
        return thresholds[np.argmax(tss_scores)], max(tss_scores)

    def train_logistic(self, inputs_profiles, labels, purpose, time1, time2):
        self.purpose = purpose
        pos_inputs, neg_inputs = get_samples(inputs_profiles, labels, self.purpose, time1, time2)
        
        self.models = []
        self.thresholds = []
        self.tss_scores = []

        # combine the data
        inputs, targets = self.combine(pos_inputs, neg_inputs)
        inputs = np.array([x.flatten() for x in inputs])
        
        X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1, random_state=42, stratify=targets)
        n_size = X_train.shape[0]
        for i in range(self.n_bootstrap):
            # bootstrap sampling
            print(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            X_boot, y_boot = resample(X_train, y_train, replace=True, n_samples= n_size, random_state=i)

            pca_logregCV = Pipeline([
                    ('scale', StandardScaler()),       
                    ('pca', PCA(n_components=0.95)),     
                    ('logreg', LogisticRegressionCV(
                            penalty='l2',
                            class_weight='balanced',
                            cv=5,
                            Cs=5,
                            scoring='roc_auc',
                            max_iter=1000,
                            n_jobs=-1))
                ])
            pca_logregCV.fit(X_boot, y_boot)
            self.models.append(pca_logregCV)

            # tss on validation set
            y_val_prob = pca_logregCV.predict_proba(X_val)[:, 1]
            best_threshold, best_tss = self.get_best_threshold(y_val, y_val_prob)
            self.thresholds.append(best_threshold)
            self.tss_scores.append(best_tss)
        
        print(f"The 95% CI for threshold is {np.percentile(self.thresholds, 2.5)} - {np.percentile(self.thresholds, 97.5)}")
        print(f"The 95% CI for TSS is {np.percentile(self.tss_scores, 2.5)} - {np.percentile(self.tss_scores, 97.5)}")

        iterations = np.arange(1, self.n_bootstrap + 1)
        mean_thresholds = np.cumsum(self.thresholds) / iterations
        mean_tss_scores = np.cumsum(self.tss_scores) / iterations
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(mean_thresholds, label='Mean Threshold')
        plt.axhline(y=np.percentile(self.thresholds, 2.5), color='r', linestyle='--', label='95% CI Lower')
        plt.axhline(y=np.percentile(self.thresholds, 97.5), color='g', linestyle='--', label='95% CI Upper')
        plt.title('Thresholds Convergence with 95% CI')
        plt.xlabel('Bootstrap Iteration')
        plt.ylabel('Threshold')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(mean_tss_scores, label='Mean TSS')
        plt.axhline(y=np.percentile(self.tss_scores, 2.5), color='r', linestyle='--', label='95% CI Lower')
        plt.axhline(y=np.percentile(self.tss_scores, 97.5), color='g', linestyle='--', label='95% CI Upper')
        plt.title('TSS Convergence with 95% CI')
        plt.xlabel('Bootstrap Iteration')
        plt.ylabel('TSS')
        plt.legend()
        plt.tight_layout()
        plt.show()