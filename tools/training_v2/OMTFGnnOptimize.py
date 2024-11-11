# OMTFGnnOptimize.py

import optuna
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.profiler
import numpy as np
import wandb
import os
import logging

from OMTFGnnModel import GNNModel
from OMTFGnnTrain import train, evaluate, save_checkpoint, load_checkpoint


def objective(trial, dataset, config):
    """
    Objective function for Optuna to optimize GNN hyperparameters using K-Fold Cross-Validation.

    Parameters:
    - trial (optuna.trial.Trial): Optuna trial object.
    - dataset (list of torch_geometric.data.Data): Graph dataset.
    - config (dict): Loaded YAML configuration.

    Returns:
    - float: Mean validation accuracy across all folds.
    """
    training_config = config['training']
    logging_config = config['logging']
    resources_config = config['resources']

    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    num_layers = trial.suggest_int('num_layers', 2, 6)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gnn_type = trial.suggest_categorical('gnn_type', ['GCN', 'GraphSAGE', 'GAT', 'GIN'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu'])
    grad_clip = trial.suggest_float('grad_clip', 0.5, 5.0, step=0.1)

    # Define output dimensions based on your task
    NUM_CLASSES = 3  # Adjust based on your dataset

    # Initialize StratifiedKFold
    n_splits = config['data_processing'].get('n_splits', 5)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config['training'].get('seed', 42))

    # Extract labels for StratifiedKFold
    labels = [data.y.argmax(dim=0).item() for data in dataset]

    # Initialize list to store validation accuracies for each fold
    val_accuracies = []

    # Initialize W&B if required
    if logging_config.get('use_wandb', False):
        wandb.init(project=logging_config.get('wandb_project', 'gnn-optimization'), reinit=True)
        wandb.config.update(trial.params)

    # Initialize Profiler if enabled
    enable_profiling = config['logging'].get('enable_profiling', False)
    if enable_profiling:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(logging_config.get('profiler_output_dir', "../logs/profiler/")),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        profiler.start()
    else:
        profiler = None

    try:
        # Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(skf.split(dataset, labels)):
            logging.info(f"Starting Fold {fold + 1}/{n_splits}")

            # Create training and validation subsets
            train_subset = [dataset[i] for i in train_idx]
            val_subset = [dataset[i] for i in val_idx]

            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=resources_config.get('num_workers', 4),
                pin_memory=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=resources_config.get('num_workers', 4),
                pin_memory=True
            )

            # Initialize the model
            model = GNNModel(
                input_dim=len(dataset[0].x[0]),
                hidden_dim=hidden_dim,
                output_dim=NUM_CLASSES,
                num_layers=num_layers,
                dropout=dropout,
                gnn_type=gnn_type
            ).to(resources_config.get('device', "cuda"))

            # Optionally add Batch Normalization layers
            if use_batch_norm:
                model.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

            # Initialize optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()

            # Initialize Learning Rate Scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )

            # Optionally set different activation functions
            if activation == 'relu':
                activation_fn = nn.ReLU()
            else:
                activation_fn = nn.LeakyReLU()

            # Initialize GradScaler for mixed precision training (optional)
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

            # Training loop for the current fold
            EPOCHS = training_config.get('epochs', 50)
            for epoch in range(EPOCHS):
                train_loss = train(
                    model, optimizer, criterion, train_loader,
                    device=resources_config.get('device', "cuda"),
                    scaler=scaler,
                    grad_clip=grad_clip,
                    profiler=profiler
                )
                val_loss, val_accuracy = evaluate(
                    model, criterion, val_loader,
                    device=resources_config.get('device', "cuda")
                )

                # Step the scheduler based on validation loss
                scheduler.step(val_loss)

                # Logging to W&B
                if logging_config.get('use_wandb', False):
                    wandb.log({
                        f'fold_{fold + 1}/epoch': epoch + 1,
                        f'fold_{fold + 1}/train_loss': train_loss,
                        f'fold_{fold + 1}/val_loss': val_loss,
                        f'fold_{fold + 1}/val_accuracy': val_accuracy,
                        f'fold_{fold + 1}/learning_rate': optimizer.param_groups[0]['lr']
                    })

                # Report intermediate objective value to Optuna
                trial.report(val_accuracy, fold * EPOCHS + epoch)

                # Handle pruning based on the intermediate value
                if trial.should_prune():
                    if logging_config.get('use_wandb', False):
                        wandb.finish()
                    raise optuna.exceptions.TrialPruned()

            # After training for all epochs in the current fold, store the validation accuracy
            val_accuracies.append(val_accuracy)
            logging.info(f"Fold {fold + 1} completed with Validation Accuracy: {val_accuracy:.4f}")

    finally:
        if profiler:
            profiler.stop()

    # Calculate mean validation accuracy across all folds
    mean_val_accuracy = np.mean(val_accuracies)
    logging.info(f"Mean Validation Accuracy across {n_splits} folds: {mean_val_accuracy:.4f}")

    # Log the mean validation accuracy to W&B
    if logging_config.get('use_wandb', False):
        wandb.log({'mean_val_accuracy': mean_val_accuracy})
        wandb.finish()

    return mean_val_accuracy


def run_optimization(graph_creation_model, config, use_wandb=False, wandb_project='gnn-optimization-project', wandb_api_key=None, enable_profiling=False):
    """
    Runs the hyperparameter optimization process using Optuna.

    Parameters:
    - graph_creation_model (GraphCreationModel): The graph creation model instance.
    - config (dict): Loaded YAML configuration.
    - use_wandb (bool): Whether to use Weights & Biases for experiment tracking.
    - wandb_project (str): W&B project name.
    - wandb_api_key (str): W&B API key.
    - enable_profiling (bool): Whether to enable PyTorch Profiler.

    Returns:
    - None
    """
    device = config['resources'].get('device', 'cuda')
    dataset = graph_creation_model.pyg_graphs

    # Initialize Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=config['training'].get('seed', 42)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, dataset, config),
        n_trials=config['training'].get('n_trials', 50)
    )

    logging.info('Number of finished trials: {}'.format(len(study.trials)))
    logging.info('Best trial:')

    trial = study.best_trial

    logging.info('  Value: {:.4f}'.format(trial.value))
    logging.info('  Params: ')
    for key, value in trial.params.items():
        logging.info(f'    {key}: {value}')

    # Save the study
    os.makedirs(config['logging'].get('checkpoints_dir', "../checkpoints/"), exist_ok=True)
    torch.save(study, os.path.join(config['logging']['checkpoints_dir'], 'optuna_study.pt'))
    logging.info(f"Optuna study saved as '{os.path.join(config['logging']['checkpoints_dir'], 'optuna_study.pt')}'.")

    # Visualize optimization history
    try:
        import optuna.visualization as vis
        fig = vis.plot_optimization_history(study)
        os.makedirs(config['logging'].get('logs_dir', "../logs/"), exist_ok=True)
        fig.write_html(os.path.join(config['logging']['logs_dir'], "optimization_history.html"))
        logging.info(f"Optimization history saved as '{os.path.join(config['logging']['logs_dir'], 'optimization_history.html')}'.")
    except ImportError:
        logging.warning("Optuna visualization module not found. Skipping visualization.")
