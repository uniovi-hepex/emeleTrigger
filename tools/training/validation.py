import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_graph_features(data_loader, output_dir='Train', label='Model',task="regression"):
    if task == "regression":
        plot_graph_feature_regression(data_loader, output_dir, label)

    elif task == "classification":
        plot_graph_feature_classification(data_loader, output_dir, label)

def plot_graph_feature_classification(data_loader, output_dir='Train', label='Model'):
    feature_names = ["eta", "phi", "R",  "deltaPhi", "deltaEta", "isMatched"]

    for batch in data_loader:
        features = batch.x.numpy()
        classification = batch.y.numpy()
        num_features = features.shape[1]
        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
        axs = axs.flatten()
        
        # Plot node features
        for i in range(num_features):
            axs[i].hist(features[:, i], bins=30, alpha=0.75)
            axs[i].set_title(f'Feature {feature_names[i]} Histogram')
            axs[i].set_xlabel(f'Feature {feature_names[i]} Value')
            axs[i].set_ylabel('Frequency')
        
        
        # plot the number of edges of each graph
        for i in range(batch.edge_attr.shape[1]):
            axs[i+num_features].hist(batch.edge_attr[:, i], bins=30, alpha=0.75)
            axs[i+num_features].set_title(f'Feature {feature_names[i+num_features]} Histogram')
            axs[i+num_features].set_xlabel(f'Feature {feature_names[i+num_features]} Value')
            axs[i+num_features].set_ylabel('Frequency')
        
        # Plot regression target
        axs[num_features + (batch.edge_attr.shape[1])].hist(classification, bins=30, alpha=0.75)
        axs[num_features + (batch.edge_attr.shape[1])].set_title(f'Classification target {feature_names[-1]}  Histogram')
        axs[num_features + (batch.edge_attr.shape[1])].set_xlabel(f'classification target {feature_names[-1]} Value')
        axs[num_features + (batch.edge_attr.shape[1])].set_ylabel('Frequency')
              
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.png'))
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.pdf'))
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.eps'))



def plot_graph_feature_regression(data_loader, output_dir='Train', label='Model'):   
    feature_names = ["eta", "phi", "R",  "deltaPhi", "deltaEta","Q/pt"]
    for batch in data_loader:
        features = batch.x.numpy()
        regression = batch.y.numpy()
        num_features = features.shape[1]
        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
        axs = axs.flatten()
        
        # Plot node features
        for i in range(num_features):
            axs[i].hist(features[:, i], bins=30, alpha=0.75)
            axs[i].set_title(f'Feature {feature_names[i]} Histogram')
            axs[i].set_xlabel(f'Feature {feature_names[i]} Value')
            axs[i].set_ylabel('Frequency')
        
        
        # plot the number of edges of each graph
        for i in range(batch.edge_attr.shape[1]):
            axs[i+num_features].hist(batch.edge_attr[:, i], bins=30, alpha=0.75)
            axs[i+num_features].set_title(f'Feature {feature_names[i+num_features]} Histogram')
            axs[i+num_features].set_xlabel(f'Feature {feature_names[i+num_features]} Value')
            axs[i+num_features].set_ylabel('Frequency')
        
        # Plot regression target
        axs[num_features + (batch.edge_attr.shape[1])].hist(regression, bins=30, alpha=0.75)
        axs[num_features + (batch.edge_attr.shape[1])].set_title(f'Regression target {feature_names[-1]}  Histogram')
        axs[num_features + (batch.edge_attr.shape[1])].set_xlabel(f'Regression target {feature_names[-1]} Value')
        axs[num_features + (batch.edge_attr.shape[1])].set_ylabel('Frequency')
              
        plt.tight_layout()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.png'))
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.pdf'))
        fig.savefig(os.path.join(output_dir, f'{label}_inputFeatures.eps'))

        break  # Only draw the first batch

@torch.no_grad()
def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_regression = []
    all_prediction = []
    for data in test_loader:
        #only one batch
        data = data.to(device)
        out = model(data)
        all_regression.append(data.y.cpu().numpy())
        all_prediction.append(out.cpu().numpy())
    
    # Concatenar todas las predicciones y valores objetivo
    all_regression = np.concatenate(all_regression, axis=0)
    all_prediction = np.concatenate(all_prediction, axis=0)
    
    return all_regression, all_prediction

def plot_prediction_results(regression, prediction, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    print("Plotting Regression target")
    axs[0].hist(regression, bins=np.arange(-0.5,0,0.006), alpha=0.75, label='Regression target')
    axs[0].hist(prediction, bins=np.arange(-0.5,0,0.006), alpha=0.75, label='Prediction')
    axs[0].set_title(f'Regression target and prediction for {model}')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].scatter(regression, prediction, alpha=0.5)
    axs[1].plot([min(prediction), max(prediction)], [min(prediction), max(prediction)], color='red', linestyle='--') # Line of equality
    axs[1].set_title(f'Regression target vs prediction for {model}')
    axs[1].set_xlabel('Regression target')
    axs[1].set_ylabel('Prediction')

    axs[2].hist(prediction - regression, bins=30, alpha=0.75)
    axs[2].set_title(f'Residuals for {model}')
    axs[2].set_xlabel('Residual')
    axs[2].set_ylabel('Frequency')
    
    # Calculate the bias and resolution and plot them in the graph
    bias = np.mean(prediction - regression)
    resolution = np.std(prediction - regression)

    # Add text box with bias and resolution
    textstr = f'Bias: {bias:.4f}\nResolution: {resolution:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[2].text(0.95, 0.95, textstr, transform=axs[2].transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.eps'))


def evaluate_model_classification(model, test_loader, device):
    model.eval()
    total_loss = 0
    all_target = []
    all_pred   = []
    for data in test_loader:
        #only one batch
        data = data.to(device)
        out = model(data)
        pred = (torch.sigmoid(out) > 0.5).long()  # Assuming binary classification with sigmoid activation
        all_target.append(data.y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    
    # Concatenar todas las predicciones y valores objetivo
    all_target = np.concatenate(all_target, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    
    return all_target, all_pred


def plot_prediction_results_classification(target, prediction, output_dir='Test', model='model', label='SaveModel'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    print("Plotting Target")
    axs[0].hist(target, bins=np.arange(-0.05,1.05,0.05), alpha=0.75, label='Target')
    axs[0].hist(prediction, bins=np.arange(-0.05,1.05,0.05), alpha=0.75, label='Prediction')
    ## Log
    axs[0].set_xlim(-0.05, 1.05)
    axs[0].set_yscale('log')
    axs[0].set_title(f'Target and prediction for {model}')
    axs[0].set_xlabel('Value')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    axs[1].scatter(target, prediction, alpha=0.5)
    axs[1].plot([min(prediction), max(prediction)], [min(prediction), max(prediction)], color='red', linestyle='--') # Line of equality
    axs[1].set_title(f'Target vs prediction for {model}')
    axs[1].set_xlabel('Target')
    axs[1].set_ylabel('Prediction')

    axs[2].hist(prediction - target, bins=30, alpha=0.75)
    axs[2].set_title(f'Residuals for {model}')
    axs[2].set_xlabel('Residual')
    axs[2].set_ylabel('Frequency')
    
    # Calculate the bias and resolution and plot them in the graph
    bias = np.mean(prediction - target)
    resolution = np.std(prediction - target)

    # Add text box with bias and resolution
    textstr = f'Bias: {bias:.4f}\nResolution: {resolution:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axs[2].text(0.95, 0.95, textstr, transform=axs[2].transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_prediction_results.eps'))

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def compute_classification_metrics(y_true, y_pred):
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    # Calcular las métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    
    # Imprimir los resultados
    print("Matriz de confusión:")
    print(cm)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    
    return cm, accuracy, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          output_dir='Test',
                          label='SaveModel',
                          title='Matriz de Confusión',
                          cmap=plt.cm.Blues):
    """
    Esta función imprime y grafica la matriz de confusión.
    La normalización se aplica si normalize=True.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{label}_confusion_matrix.png'))
    fig.savefig(os.path.join(output_dir, f'{label}_confusion_matrix.pdf'))
    fig.savefig(os.path.join(output_dir, f'{label}_confusion_matrix.eps'))


