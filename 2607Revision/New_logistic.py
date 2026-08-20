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
    def __init__(self, n_booststrap=30, pos_weight=None):
        self.n_bootstrap = n_booststrap
        self.models = []
        self.purpose = None
        self.pos_weight = pos_weight

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
        """
    Train bootstrap PCA-logistic-regression models.

        Training modes
        --------------
        self.pos_weight is None:
            Use all available positive and negative training samples.
            Bootstrap the two classes separately using their original counts.

        self.pos_weight is a positive number, e.g. 2:
            For each bootstrap iteration, sample

                n_negative = self.pos_weight * n_positive

            negative observations.

            LogisticRegressionCV(class_weight="balanced") then calculates
            class weights from the actual sampled class counts. For a 2:1
            negative-to-positive ratio, the positive observations receive twice
            the per-observation weight of the negative observations.

        Notes
        -----
        The validation set is not undersampled. It retains the natural class
        distribution.
        """
        self.purpose = purpose
        pos_inputs, neg_inputs = get_samples(inputs_profiles, labels, self.purpose, time1, time2)

        if len(pos_inputs) == 0:
            raise ValueError("No positive samples were found.")

        if len(neg_inputs) == 0:
            raise ValueError("No negative samples were found.")

        # combine the data
        inputs, targets = self.combine(pos_inputs, neg_inputs)
        inputs = np.array([x.flatten() for x in inputs], dtype=np.float32)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    
        self.models = []
        self.thresholds = []
        self.tss_scores = []

    

        X_train, X_val, y_train, y_val = train_test_split(inputs, targets, test_size=0.1, random_state=42, stratify=targets)

        # Separate the training data by class
        positive_mask = y_train == 1
        negative_mask = y_train == 0

        X_pos = X_train[positive_mask]
        y_pos = y_train[positive_mask]

        X_neg = X_train[negative_mask]
        y_neg = y_train[negative_mask]

        n_pos = len(X_pos)
        n_neg = len(X_neg)

        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                "The training split must contain both classes."
            )

        if self.pos_weight is None:
            negative_sampling_ratio = n_neg / n_pos

            print(
                "Using all training samples: "
                f"{n_pos} positives and {n_neg} negatives."
            )
        else:
            if not isinstance(
                self.pos_weight,
                (int, float, np.integer, np.floating)
            ):
                raise TypeError(
                    "self.pos_weight must be None or a positive number."
                )

            if self.pos_weight <= 0:
                raise ValueError(
                    "self.pos_weight must be greater than zero."
                )

            negative_sampling_ratio = float(self.pos_weight)

            requested_n_neg = int(
                round(negative_sampling_ratio * n_pos)
            )

            print(
                "Using negative sampling with "
                f"{negative_sampling_ratio:g} negatives per positive."
            )

            print(
                f"Each bootstrap will use {n_pos} positives and "
                f"{min(requested_n_neg, n_neg)} negatives."
            )
        
        for i in range(self.n_bootstrap):
            # bootstrap sampling
            print(f"Bootstrap iteration {i+1}/{self.n_bootstrap}")
            X_pos_boot, y_pos_boot = resample(X_pos, y_pos, replace=True, n_samples= n_pos, random_state=i)
            if self.pos_weight is None:
                # Bootstrap all negatives using the original negative count
                X_neg_boot, y_neg_boot = resample(
                    X_neg,
                    y_neg,
                    replace=True,
                    n_samples=n_neg,
                    random_state=10_000 + i
                )

            else:
                requested_n_neg = int(
                    round(
                        negative_sampling_ratio
                        * len(X_pos_boot)
                    )
                )

                replace_negatives = requested_n_neg > n_neg

                X_neg_boot, y_neg_boot = resample(
                    X_neg,
                    y_neg,
                    replace=replace_negatives,
                    n_samples=requested_n_neg,
                    random_state=10_000 + i
                )

            # Combine the class-wise bootstrap samples
            X_boot = np.concatenate(
                [X_pos_boot, X_neg_boot],
                axis=0
            )
            y_boot = np.concatenate([y_pos_boot, y_neg_boot], axis=0)

            # Shuffle the complete bootstrap sample
            rng = np.random.default_rng(seed=20_000 + i)

            indices = rng.permutation(len(y_boot))

            X_boot = X_boot[indices]
            y_boot = y_boot[indices]

            # Record actual class counts
            n_pos_boot = int(np.sum(y_boot == 1))

            n_neg_boot = int(np.sum(y_boot == 0))

            actual_ratio = (n_neg_boot / n_pos_boot)

            print(
                f"Bootstrap sample: "
                f"{n_pos_boot} positives, "
                f"{n_neg_boot} negatives; "
                f"ratio={actual_ratio:.4f}"
            )


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
                            n_jobs=1,
                            random_state= 100 + i))
                ])
            pca_logregCV.fit(X_boot, y_boot)
            self.models.append(pca_logregCV)

            # tss on validation set
            y_val_prob = pca_logregCV.predict_proba(X_val)[:, 1]
            best_threshold, best_tss = self.get_best_threshold(y_val, y_val_prob)
            self.thresholds.append(best_threshold)
            self.tss_scores.append(best_tss)
        
        print(f"The 95% CI for threshold is {np.percentile(self.thresholds, 2.5)} - {np.percentile(self.thresholds, 97.5)}")
        print(f"The chosen threshold: {np.mean(self.thresholds)}")
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