import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score, precision_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

def plot_node_feature_histograms(data_loader, output_dir='Train', label='Model', step='AfterNormalization', node_feature_labels = ["EtaG", "CosPhi", "SinPhi", "R", "PhiG", "Layer", "Type"]):
    # step = Input, AfterNormalization
    for batch in data_loader:
        features = batch.x.numpy()
        num_features = features.shape[1]
        num_cols = (num_features + 1) // 2
        fig, axs = plt.subplots(2, num_cols, figsize=(15, 15))
        axs = axs.flatten()
        
        # Plot node features
        for i in range(num_features):
            axs[i].hist(features[:, i], bins=30, alpha=0.75)
            axs[i].set_title(f'Node Feature {node_feature_labels[i]} Histogram')
            axs[i].set_xlabel(f'Node Feature {node_feature_labels[i]} Value')
            axs[i].set_ylabel('Frequency')
            
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{label}_{step}NodeFeatures.png'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}NodeFeatures.pdf'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}NodeFeatures.eps'))
    
        break  # Only draw the first batch

def plot_edge_attr_histograms(data_loader, output_dir='Train', label='Model', step='AfterNormalization'):
    # step = Input, AfterNormalization
    edge_attr_labels = ["deltaPhi", "deltaEta", "deltaR"]
    for batch in data_loader:
        edge_attr = batch.edge_attr
        num_edge_attr = edge_attr.shape[1]
        fig, axs = plt.subplots(1, num_edge_attr, figsize=(12, 3))
        axs = axs.flatten()
        
        # plot the number of edges of each graph
        for i in range(edge_attr.shape[1]):
            axs[i].hist(edge_attr[:, i], bins=30, alpha=0.75)
            axs[i].set_title(f'Edge Attr {edge_attr_labels[i]} Histogram')
            axs[i].set_xlabel(f'Edge Attr {edge_attr_labels[i]} Value')
            axs[i].set_ylabel('Frequency')
              
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{label}_{step}EdgeAttr.png'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}EdgeAttr.pdf'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}EdgeAttr.eps'))

        break  # Only draw the first batch

def plot_Nodes_and_Edges_histograms(data_loader, output_dir='Train', label='Model', step='AfterNormalization'):
    # step = Input, AfterNormalization
    for batch in data_loader:
        num_nodes = batch.x.shape[0]
        num_edges = batch.edge_index.shape[1]
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs = axs.flatten()

        axs[0].hist(num_nodes, bins=30, alpha=0.75)
        axs[0].set_title('Number of Nodes per Graph Histogram')
        axs[0].set_xlabel('Number of Nodes per Graph')
        axs[0].set_ylabel('Frequency')

        axs[1].hist(num_edges, bins=30, alpha=0.75)
        axs[1].set_title('Number of Edges per Graph Histogram')
        axs[1].set_xlabel('Number of Edges per Graph')
        axs[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{label}_{step}_Nodes_and_Edges_per_Graph.png'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}_Nodes_and_Edges_per_Graph.pdf'))
        fig.savefig(os.path.join(output_dir, f'{label}_{step}_Nodes_and_Edges_per_Graph.eps'))

        break  # Only draw the first batch

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    for data in test_loader:
        #only one batch
        data = data.to(device)
        logits = model(data.x, data.edge_index, data.edge_attr).view(-1)
        labels = data.edge_label.view(-1).float()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenar todas las predicciones y valores objetivo
    y_pred = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    
    return y_pred, y_true

def metric_scores_file(y_pred, y_true, metrics, threshold_classification=0.7, output_dir='Test', labels_classification=[0, 1]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metrics_file = os.path.join(output_dir, 'metrics.txt')

    all_metrics_score = {}
    y_pred_discrete = (torch.sigmoid(torch.tensor(y_pred)).numpy() >= threshold_classification).astype(int)
    if metrics == "AllMetrics":
        fpr, tpr, thresholds = roc_curve(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        auc_score = roc_auc_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        GINI_score = 2*auc_score-1
        f1_metric = f1_score(y_true, y_pred_discrete)
        recall_metric = recall_score(y_true, y_pred_discrete)
        precision_metric = precision_score(y_true, y_pred_discrete)
        precision_per_class = precision_score(y_true, y_pred_discrete, average=None, labels=labels_classification)
        ap_metric = average_precision_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        confusion_mtrx = confusion_matrix(y_true, y_pred_discrete)
        TN = confusion_mtrx[0, 0]
        FP = confusion_mtrx[0, 1]
        FN = confusion_mtrx[1, 0]
        TP = confusion_mtrx[1, 1]
        efficiency_score = accuracy_score(y_true, y_pred_discrete)
        all_metrics_score["AUC"] = auc_score
        all_metrics_score["GINI"] = GINI_score
        all_metrics_score["F1"] = f1_metric
        all_metrics_score["Recall"] = recall_metric
        all_metrics_score["Precision"] = precision_metric
        all_metrics_score["Precision_per_Class"] = precision_per_class
        all_metrics_score["AP"] = ap_metric
        all_metrics_score["TN"] = TN
        all_metrics_score["FP"] = FP
        all_metrics_score["FN"] = FN
        all_metrics_score["TP"] = TP
        all_metrics_score["Efficiency"] = efficiency_score
    if metrics == "ROC_AUC_GINI":
        fpr, tpr, thresholds = roc_curve(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        auc_score = roc_auc_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        GINI_score = 2*auc_score-1
        all_metrics_score["AUC"] = auc_score
        all_metrics_score["GINI"] = GINI_score
    if metrics == "F1":
        f1_metric = f1_score(y_true, y_pred_discrete)
        all_metrics_score["F1"] = f1_metric
    if metrics == "Recall_Precision_AP":
        recall_metric = recall_score(y_true, y_pred_discrete)
        precision_metric = precision_score(y_true, y_pred_discrete)
        precision_per_class = precision_score(y_true, y_pred_discrete, average=None, labels=labels_classification)
        ap_metric = average_precision_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
        all_metrics_score["Recall"] = recall_metric
        all_metrics_score["Precision"] = precision_metric
        all_metrics_score["Precision_per_Class"] = precision_per_class
        all_metrics_score["AP"] = ap_metric
    if metrics == "Confusion_Matrix":
        confusion_mtrx = confusion_matrix(y_true, y_pred_discrete)
        TN = confusion_mtrx[0, 0]
        FP = confusion_mtrx[0, 1]
        FN = confusion_mtrx[1, 0]
        TP = confusion_mtrx[1, 1]
        all_metrics_score["TN"] = TN
        all_metrics_score["FP"] = FP
        all_metrics_score["FN"] = FN
        all_metrics_score["TP"] = TP
    if metrics == "Efficiency":
        efficiency_score = accuracy_score(y_true, y_pred_discrete)
        all_metrics_score["Efficiency"] = efficiency_score

    with open(metrics_file, 'w') as f:
        for metric_name, metric_score in all_metrics_score.items():
            f.write(f"{metric_name} = {metric_score}\n")
    

def plot_ROC_curve(y_pred, y_true, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fpr, tpr, thresholds = roc_curve(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
    auc_score = roc_auc_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
    
    fig, axs = plt.subplots(figsize=(15, 15))

    print("Plotting ROC Curve")
    axs.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    axs.plot([0, 1], [0, 1], 'r--')  # línea de base
    axs.set_title(f'ROC Curve for {model}')
    axs.set_xlabel('False Positive Rate')
    axs.set_ylabel('True Positive Rate')
    axs.legend(loc='lower right')
    axs.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_Curve.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_Curve.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_Curve.eps'))

def plot_prec_vs_rec(y_pred, y_true, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    precision, recall, _ = precision_recall_curve(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())
    ap_score = average_precision_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy())


    fig, axs = plt.subplots(figsize=(15, 15))

    print("Plotting Precision vs Recall Curve")
    axs.plot(recall, precision, label=f'AP = {ap_score:.3f}')
    axs.set_title(f'Precision vs Recall Curve for {model}')
    axs.set_xlabel('Recall')
    axs.set_ylabel('Precision')
    axs.legend(loc='lower left')
    axs.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_Recall_Curve.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_Recall_Curve.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_Recall_Curve.eps'))
    
def plot_predicted_results(y_pred, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fig, axs = plt.subplots(figsize=(15, 15))

    print("Plotting Predicted Scores")
    axs.hist(torch.sigmoid(torch.tensor(y_pred)).numpy(), bins=50, alpha=0.7)
    axs.set_title(f'Distribution of Predicted Scores for {model}')
    axs.set_xlabel('Predicted Probability')
    axs.set_ylabel('Count')

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_of_Predicted_Scores.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_of_Predicted_Scores.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_of_Predicted_Scores.eps'))

def plot_prec_per_class(y_pred, y_true, output_dir='Test', model='model', threshold_classification=0.7, label='SaveModel', labels_classification=[0, 1]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    y_pred_discrete = (torch.sigmoid(torch.tensor(y_pred)).numpy() >= threshold_classification).astype(int)
    precisions = precision_score(y_true, y_pred_discrete, average=None, labels=labels_classification)
    
    fig, axs = plt.subplots(figsize=(15, 15))

    print("Plotting Precision per Class")
    axs.bar(labels_classification, precisions)
    axs.set_title(f'Precision per Class for {model}')
    axs.set_xlabel('Classification Labels')
    axs.set_ylabel('Precision')
    axs.legend(loc='lower left')
    axs.set_ylim(0, 1)
    axs.set_xticks(labels_classification)
    axs.grid(True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_per_Class.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_per_Class.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_Precision_per_Class.eps'))

def evaluate_per_event(model, loader, device, threshold_classification=0.7):
    model.eval()
    results = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index, data.edge_attr).view(-1).cpu().numpy()
        logits_discrete = (logits >= threshold_classification).astype(int)
        labels = data.edge_label.view(-1).cpu().numpy()
        auc = roc_auc_score(labels, logits) if len(np.unique(labels)) > 1 else np.nan
        eff = accuracy_score(labels, logits_discrete) if len(np.unique(labels)) > 1 else np.nan
        results.append({
            'muon_pt': float(data.muon_vars[3]),        # variable de evento
            'auc': auc,
            'efficiency': eff,
            'n_edges': len(labels)
        })
    return results

def summarize_by_variable(results, key = 'muon_pt', bins = [-200,-100,0,100,200,300,400,500,600]):
    pts = np.array([r[key] for r in results])
    aucs = np.array([r['auc'] for r in results])

    means, stds, centers = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pts >= lo) & (pts < hi)
        if np.sum(mask) < 5: 
            means.append(np.nan); stds.append(np.nan)
        else:
            vals = aucs[mask]
            means.append(np.nanmean(vals)); stds.append(np.nanstd(vals))
        centers.append((lo+hi)/2)
    return np.array(centers), np.array(means), np.array(stds)

def summarize_by_variable_efficiency(results, key='muon_pt', bins=[-200,-100,0,100,200,300,400,500,600]):
    pts = np.array([r[key] for r in results])
    effs = np.array([r['efficiency'] for r in results])

    means, stds, centers = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pts >= lo) & (pts < hi)
        if np.sum(mask) < 5: 
            means.append(np.nan); stds.append(np.nan)
        else:
            vals = effs[mask]
            means.append(np.nanmean(vals)); stds.append(np.nanstd(vals))
        centers.append((lo+hi)/2)
    return np.array(centers), np.array(means), np.array(stds)

def plot_auc_vs(results, key = 'muon_pt', bins = [-200,-100,0,100,200,300,400,500,600], output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    centers, means, stds = summarize_by_variable(results, key, bins)

    fig, axs = plt.subplots(figsize=(15, 15))

    print(f"Plotting AUC vs {key} Curve")
    axs.errorbar(centers, means, yerr=stds, fmt='-o')
    axs.set_title(f'AUC vs {key}')
    axs.set_xlabel(key)
    axs.set_ylabel('AUC')
    axs.grid(True) 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_AUC_vs_{key}.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_AUC_vs_{key}.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_AUC_vs_{key}.eps'))

def plot_roc_by_bins(y_true, y_score, var, name, bins, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    centers, means, stds = summarize_by_variable(results, key, bins)

    fig, axs = plt.subplots(figsize=(15, 15))

    print(f"Plotting ROC curves by {name}")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (var >= lo) & (var < hi)
        if mask.sum() < 10: continue
        fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
        auc_bin = roc_auc_score(y_true[mask], y_score[mask])
        axs.plot(fpr, tpr, label=f"{name} ∈ [{lo},{hi}) AUC={auc_bin:.2f}")
    
    axs.plot([0,1], [0,1], 'k--')
    axs.set_title(f"ROC curves by {name}")
    axs.set_xlabel("False Positive Rate")
    axs.set_ylabel("True Positive Rate")
    axs.grid(True) 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_by_bins_{name}.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_by_bins_{name}.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_ROC_by_bins_{name}.eps'))

def plot_GINI_vs(results, key = 'muon_pt', bins = [-200,-100,0,100,200,300,400,500,600], output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    centers, means, stds = summarize_by_variable(results, key, bins)
    mean_GINI_idx = [2*mean-1 for mean in means]
    stds_GINI_IDX = [2*std for std in stds]

    fig, axs = plt.subplots(figsize=(15, 15))

    print(f"Plotting AUC vs {key} Curve")
    axs.errorbar(centers, mean_GINI_idx, yerr=stds_GINI_IDX, fmt='-o')
    axs.set_title(f'GINI index  vs {key}')
    axs.set_xlabel(key)
    axs.set_ylabel('GINI index ')
    axs.grid(True) 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_GINI_index_vs_{key}.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_GINI_index_vs_{key}.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_GINI_index_vs_{key}.eps'))

def plot_efficiency_vs(results, key = 'muon_pt', bins = [-200,-100,0,100,200,300,400,500,600], output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    centers, means, stds = summarize_by_variable_efficiency(results, key, bins)
    
    fig, axs = plt.subplots(figsize=(15, 15))

    print(f"Plotting Efficiency vs {key} Curve")
    axs.errorbar(centers, means, yerr=stds, fmt='-o')
    axs.set_title(f'Efficiency vs {key}')
    axs.set_xlabel(key)
    axs.set_ylabel('Efficiency')
    axs.grid(True) 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_Efficiency_vs_{key}.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_Efficiency_vs_{key}.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_Efficiency_vs_{key}.eps'))

def plot_muon_pT(results, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    muon_pts = np.array([r['muon_pt'] for r in results])
    
    fig, axs = plt.subplots(figsize=(15, 15))

    print("Distribution of Muon pT per Event")
    axs.hist(muon_pts, bins=30, edgecolor='black', alpha=0.7)
    axs.set_title('Distribution of Muon pT per Event')
    axs.set_xlabel(r'$\mu_{pT}$')
    axs.set_ylabel('Events')
    axs.grid(True) 

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_muon_pT_per_event.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_muon_pT_per_event.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_Distribution_muon_pT_per_event.eps'))
