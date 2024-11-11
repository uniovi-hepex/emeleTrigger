
# scripts/analyze_study.py

# scripts/analyze_study.py

import torch
import optuna
import optuna.visualization as vis

def main():
    # Load the Optuna study
    study = torch.load('optuna_study.pt')

    # Print best trial
    print('Best trial:')
    trial = study.best_trial

    print('  Value: {:.4f}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Plot optimization history
    fig = vis.plot_optimization_history(study)
    fig.show()

    # Plot parameter importance
    fig = vis.plot_param_importances(study)
    fig.show()

if __name__ == "__main__":
    main()
