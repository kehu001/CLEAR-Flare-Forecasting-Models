import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from New_SampleConstruction import *
from New_lstm2 import *
from utilities import *


def GenMetrics_lstm(
    model_name,
    sample_obj,
    time1,
    time2,
    model_dir,
    purpose,
    threshold=0.5,
    batch_size=2048,
    n_models=30,
    num_workers=0
):
    """
    Evaluate multiple saved LSTM bootstrap models using batched inference.

    Parameters
    ----------
    batch_size : int
        Number of samples evaluated at once. Increase until GPU memory
        utilization is reasonably high without causing out-of-memory errors.
    n_models : int
        Number of bootstrap models to evaluate.
    num_workers : int
        Number of DataLoader workers. On Windows, 0 is often the safest.
    """

    # ---------------------------------------------------------
    # Prepare test samples
    # ---------------------------------------------------------
    te_pos_inputs, te_neg_inputs = get_samples(
        sample_obj.inputs_profile,
        sample_obj.labels,
        purpose,
        time1,
        time2
    )

    print(
        f"Positive: {te_pos_inputs.shape[0]}, "
        f"Negative: {te_neg_inputs.shape[0]}"
    )
    pos_rate = te_pos_inputs.shape[0] / (te_pos_inputs.shape[0] + te_neg_inputs.shape[0]
    )

    te_inputs, te_targets = combine(
        te_pos_inputs,
        te_neg_inputs
    )

    # Avoid an unnecessary explicit copy unless normalize2 requires it.
    te_inputs = normalize2(
        te_inputs,
        te_inputs
    )

    te_inputs = np.asarray(
        te_inputs,
        dtype=np.float32
    )

    te_targets = np.asarray(
        te_targets,
        dtype=np.int64
    ).reshape(-1)

    # ---------------------------------------------------------
    # DataLoader for batched inference
    # ---------------------------------------------------------
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(te_inputs)
    )

    use_cuda = (
        isinstance(DEVICE, torch.device)
        and DEVICE.type == "cuda"
    ) or str(DEVICE).startswith("cuda")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False
    )

    # ---------------------------------------------------------
    # Allocate metric arrays
    # ---------------------------------------------------------
    pods = np.empty(n_models, dtype=np.float64)
    fars = np.empty(n_models, dtype=np.float64)
    tsss = np.empty(n_models, dtype=np.float64)
    hsss = np.empty(n_models, dtype=np.float64)
    f1s = np.empty(n_models, dtype=np.float64)

    input_dim = te_inputs.shape[2]

    # ---------------------------------------------------------
    # Evaluate all bootstrap models
    # ---------------------------------------------------------
    for i in range(n_models):

        path = os.path.join(
            model_dir,
            f"{model_name}{i}.pth"
        )

        state_dict = torch.load(
            path,
            map_location=DEVICE,
            weights_only=True
        )

        model = lstm(input_dim).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()

        predictions = []

        with torch.inference_mode():
            for (batch_inputs,) in test_loader:
                batch_inputs = batch_inputs.to(
                    DEVICE,
                    dtype=torch.float32,
                    non_blocking=use_cuda
                )

                logits = model(batch_inputs).reshape(-1)

                # Sigmoid is unnecessary for thresholding if the probability
                # threshold is converted to the equivalent logit threshold.
                if 0.0 < threshold < 1.0:
                    logit_threshold = np.log(
                        threshold / (1.0 - threshold)
                    )

                    batch_pred = (
                        logits > logit_threshold
                    ).to(torch.uint8)
                elif threshold <= 0.0:
                    batch_pred = torch.ones_like(
                        logits,
                        dtype=torch.uint8
                    )
                else:
                    batch_pred = torch.zeros_like(
                        logits,
                        dtype=torch.uint8
                    )

                predictions.append(
                    batch_pred.cpu()
                )

        y_pred_i = torch.cat(
            predictions
        ).numpy()

        tsss[i] = TSS(y_pred_i, te_targets)
        pods[i] = POD(y_pred_i, te_targets)
        fars[i] = FAR(y_pred_i, te_targets)
        hsss[i] = HSS(y_pred_i, te_targets)
        f1s[i] = F1(y_pred_i, te_targets)

    # ---------------------------------------------------------
    # Report results
    # ---------------------------------------------------------
    def print_summary(name, values):
        print(
            f"{name}: "
            f"{values.mean():.2f}"
            f"[{values.min():.2f},"
            f"{values.max():.2f}]"
        )

    print_summary("TSS", tsss)
    print_summary("HSS", hsss)
    print_summary("POD", pods)
    print_summary("F1", f1s)
    print_summary("FAR", fars)

    return tsss, hsss, pods, f1s, fars, pos_rate



def plot_metric_def(model_name1, obj1, dir1,
                     model_name2, obj2, dir2,
                     label1, label2,
                     purpose,
                     figure_name,
                     lstm = True):
    time1_list = ('2020-01-01', '2022-01-01', '2023-01-01', '2020-01-01')
    time2_list = ('2022-01-01', '2023-01-01','2025-01-01', '2025-01-01')
    period_labels = ['2020–2021', '2022', '2023–2024', '2020–2024']

    # boxplot data
    tss_box1 = []; hss_box1 = []; pod_box1 = []; f1_box1 = []; far_box1 = []; acc_box1 = []
    tss_box2 = []; hss_box2 = []; pod_box2 = []; f1_box2 = []; far_box2 = []; acc_box2 = []
    pos_rate_box1 = []; pos_rate_box2 = []

    for time1, time2 in zip(time1_list, time2_list):
        print(f"Time period: {time1} to {time2}")
        print("Model 1:")
        if lstm:
            tss1, hss1, pod1, f11, far1, acc1, pos_rate1 = GenMetrics_lstm(model_name1,obj1,time1,time2,dir1,purpose)
        else:
            tss1, hss1, pod1, f11, far1, acc1, pos_rate1 = GenMetrics_logreg(model_name1, obj1, purpose, time1, time2, dir1)
        tss_box1.append(tss1); hss_box1.append(hss1); pod_box1.append(pod1)
        f1_box1.append(f11); far_box1.append(far1); acc_box1.append(acc1)
        pos_rate_box1.append(pos_rate1)

        print("Model 2:")
        if lstm:
            tss2, hss2, pod2, f12, far2, acc2, pos_rate2 =GenMetrics_lstm(model_name1,obj1,time1,time2,dir1,purpose)
        else:
            tss2, hss2, pod2, f12, far2, acc2, pos_rate2 = GenMetrics_logreg(model_name2, obj2, purpose, time1, time2, dir2)
        tss_box2.append(tss2); hss_box2.append(hss2); pod_box2.append(pod2)
        f1_box2.append(f12); far_box2.append(far2); acc_box2.append(acc2)
        pos_rate_box2.append(pos_rate2)

    # ---- Plotting ----
    metrics = ['TSS', 'HSS', 'POD', 'F1', 'FAR', 'ACC']
    box1 = [tss_box1, hss_box1, pod_box1, f1_box1, far_box1, acc_box1]
    box2 = [tss_box2, hss_box2, pod_box2, f1_box2, far_box2, acc_box2]

    # ---- Save raw bootstrap values separately for Model 1 ----
    rows_model1 = []
    for i, metric in enumerate(metrics):
        for j, period in enumerate(time1_list):
            row = {
                "Metric": metric,
                "Period": f"{time1_list[j]} to {time2_list[j]}"
            }
            vals = np.array(box1[i][j])
            for k, v in enumerate(vals):
                row[f"Sample_{k+1}"] = v
            rows_model1.append(row)

    df_model1 = pd.DataFrame(rows_model1)
    #path1 = Path(f"./results/NEW_raw_metrics_{label1}.csv")
    #df_model1.to_csv(path1, index=False)

    # ---- Save raw bootstrap values separately for Model 2 ----
    rows_model2 = []
    for i, metric in enumerate(metrics):
        for j, period in enumerate(time1_list):
            row = {
                "Metric": metric,
                "Period": f"{time1_list[j]} to {time2_list[j]}"
            }
            vals = np.array(box2[i][j])
            for k, v in enumerate(vals):
                row[f"Sample_{k+1}"] = v
            rows_model2.append(row)

    df_model2 = pd.DataFrame(rows_model2)
    #path2 = Path(f"./results/NEW_raw_metrics_{label2}.csv")
    #df_model2.to_csv(path2, index=False)

    # positions: for 4 periods, each has 2 boxes
    # e.g. period 0 → x = 1,2 ; period 1 → x = 4,5 ; period 2 → x = 7,8 ; period 3 → x = 10,11
    n_periods = len(time1_list)
    base_gap = 3.0
    pos_model1 = [i * base_gap + 1 for i in range(n_periods)]
    pos_model2 = [i * base_gap + 2 for i in range(n_periods)]
    all_positions = []
    for p1, p2 in zip(pos_model1, pos_model2):
        all_positions.extend([p1, p2])
    # x locations for period centers (for xticks and pos rate lines)
    period_centers = [(p1 + p2) / 2.0 for p1, p2 in zip(pos_model1, pos_model2)]

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    color1 = 'tab:blue'
    color2 = 'tab:orange'

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # prepare boxplot data interleaving model1/model2 per period
        data = []
        for j in range(n_periods):
            data.append(box1[i][j])  # model 1 for this period
            data.append(box2[i][j])  # model 2 for this period

        bp = ax.boxplot(
            data,
            positions=all_positions,
            widths=0.7,
            patch_artist=True,
            manage_ticks=False,
            showmeans=True
        )

        # color boxes: even indices → model1, odd → model2
        for k, patch in enumerate(bp['boxes']):
            if k % 2 == 0:  # model1
                patch.set_facecolor(color1)
                patch.set_alpha(0.6)
            else:           # model2
                patch.set_facecolor(color2)
                patch.set_alpha(0.6)

        # Whisker & cap colors (optional)
        for k in range(len(bp['boxes'])):
            col = color1 if k % 2 == 0 else color2
            for wh in [bp['whiskers'][2*k], bp['whiskers'][2*k+1]]:
                wh.set_color(col)
            bp['caps'][2*k].set_color(col)
            bp['caps'][2*k+1].set_color(col)
            bp['medians'][k].set_color('black')

        ax.set_ylabel(metric)

        # Right axis for positive rate
        ax2 = ax.twinx()
        # make sure pos rate is scalar; if array, take mean
        def to_scalar(x):
            x = np.asarray(x)
            return float(x.mean()) if x.size > 1 else float(x)

        y1 = [to_scalar(v) for v in pos_rate_box1]
        y2 = [to_scalar(v) for v in pos_rate_box2]

        ax2.plot(period_centers, y1, linestyle='--', marker='o', color=color1, alpha=0.7)
        ax2.plot(period_centers, y2, linestyle='--', marker='s', color=color2, alpha=0.7)
        ax2.set_ylim(0, 1)
        #if i % 2 != 0:
        ax2.set_ylabel('Positive sample ratio')

        # Only put legend on the first subplot to avoid clutter
        if i == 0:
            # use colors from boxplots for legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color1, edgecolor='k', label=label1, alpha=0.6),
                Patch(facecolor=color2, edgecolor='k', label=label2, alpha=0.6)
            ]
            ax2.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # x-axis ticks only on bottom row
        ax.set_xticks(period_centers)
        if i >= 4:  # last row
            ax.set_xticklabels(period_labels, rotation=0, fontsize=10)
        else:
            ax.set_xticklabels([])

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.show()
    # save figure in pdf
    #os.makedirs("./figures", exist_ok=True)
    fig.savefig(f"D:\\2024_S1\\ML_SEP_2402\\Final_update\\Open_Repo/figures/{figure_name}.pdf", dpi=300)