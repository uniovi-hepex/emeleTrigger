import torch
import os,sys

from TrainModelFromGraph import TrainModelFromGraph
import matplotlib.pyplot as plt
import numpy as np

class PlotRegression:
    def __init__(self, model, test_loader):
        self.trained_model = model
        self.test_loader = test_loader
        self.pt_pred_arr = []
        self.pt_truth_arr = []
        self.MSE = 0.0
        self.resSTD = 0.0
        self.MaxErr = 0.0
        self.R2 = 0.0
        self.KL = 0.0
        self.pt_resolution = []

    def evaluate(self):
        with torch.no_grad():
            for data in self.test_loader:
                out = self.trained_model(data)
                for item in range(0, out.size(0)):
                    vector_pred = out[item]
                    vector_real = data[item].y
                    self.pt_pred_arr.append(vector_pred.item())
                    self.pt_truth_arr.append(vector_real.item())

    def eval_metrics(self):
        pred = np.array(self.pt_pred_arr, dtype='float32')
        truth = np.array(self.pt_truth_arr, dtype='float32')
        
        residuals = pred - truth
        self.MSE = (residuals**2).mean()
        self.resSTD = (residuals**2).std()
        self.MaxErr = residuals.max()
        self.R2 = 1- ((residuals**2).sum()/((pred-pred.mean())**2).sum())
        norm_prediction = pred / pred.sum()
        norm_regression = truth / truth.sum()      
        KL = np.array([(norm_prediction[i]*np.log(norm_prediction[i]/norm_regression[i])).item() for i in range(len(norm_prediction.ravel())) if norm_prediction[i]*norm_regression[i]>0 ])
        self.KL = KL.sum()
        self.pt_resolution = residuals / truth

    def plot_regression(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.clf()
        print(f"Plotting regression in {output_dir}")
        plt.hist(self.pt_truth_arr, bins=100, color='skyblue', alpha=0.5, label="truth")
        plt.hist(self.pt_pred_arr, bins=100, color='g', alpha=0.5, label="prediction")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression.png"))
        plt.clf()

        print(f"Plotting scatter in {output_dir}")
        plt.plot(self.pt_truth_arr, self.pt_pred_arr, 'o')
        plt.xlabel("Truth")
        plt.ylabel("Prediction")
        plt.savefig(os.path.join(output_dir, "pt_regression_scatter.png"))
        plt.clf()

        print(f"Plotting difference in {output_dir}")
        # plot difference between truth and prediction
        diff = [x - y for x, y in zip(self.pt_truth_arr, self.pt_pred_arr)]
        plt.hist(diff, bins=100, color='r', alpha=0.5, label="difference")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_regression_diff.png"))
        plt.clf()


    def store_metrics(self, output_dir):
        metrics = f"Mean Squared Error \t Variance of residuals \t Maximum Error \t R-squared \t Kullback-Leibler divergence \n {self.MSE} \t {self.resSTD} \t {self.MaxErr} \t {self.R2} \t {self.KL} \n"
        with open(os.path.join(output_dir, "Metrics.tab"),"w") as outfile:
            outfile.write(metrics)
        print(f"Plotting resolution in {output_dir}")
        plt.clf()
        plt.hist(self.pt_resolution, bins=100, color='b', alpha=0.5, label="predicted-target / target")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "pt_resolution.png"))
        plt.clf()

def main():

    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate GAT model")
    parser = TrainModelFromGraph.add_args(parser)
    #parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output directory for evaluation results')
    args = parser.parse_args()

    trainer = TrainModelFromGraph(**vars(args))

    trainer.load_data()
    test_loader = trainer.test_loader

    # Inicializar el modelo
    trainer.initialize_model()

    # Cargar el modelo con map_location
    if torch.cuda.is_available():
        state_dict = torch.load(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))

    # Reasignar los parámetros al modelo actual
    trainer.model.load_state_dict(state_dict)
            
    evaluator = PlotRegression(trainer.model, test_loader)
    evaluator.evaluate()
    evaluator.eval_metrics()
    evaluator.plot_regression(output_dir=args.output_dir)
    evaluator.store_metrics(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
