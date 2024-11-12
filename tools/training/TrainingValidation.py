from TrainModelFromGraph import TrainModelFromGraph

class PlotRegression:
    def __init__(self, model, test_loader, batch_size):
        self.model = model
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.pt_pred_arr = []
        self.pt_truth_arr = []

    def evaluate(self):
        with torch.no_grad():
            for data in self.test_loader:
                out = self.model(data)
                for item in range(0, out.size(0)):
                    vector_pred = out[item]
                    vector_real = data[item].y
                    self.pt_pred_arr.append(vector_pred.item())
                    self.pt_truth_arr.append(vector_real.item())

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

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate GAT model")
    parser.add_argument('--graph_path', type=str, default='graph_folder', help='Path to the graph data')
    parser.add_argument('--out_path', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output path for the results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--model', type=str, default='GAT', help='Model to use for training')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--model_path', type=str, default='Bsize_gmp_64_lr5e-4_v3/model_1000.pth', help='Path to the saved model for evaluation')
    parser.add_argument('--output_dir', type=str, default='Bsize_gmp_64_lr5e-4_v3', help='Output directory for evaluation results')
    args = parser.parse_args()

    trainer = TrainModelFromGraph(**vars(args))

    # For evaluating:
    trainer.load_data()
    test_loader = trainer.test_loader
    model = torch.load(args.model_path)
            
    evaluator = PlotRegresson(model, test_loader, batch_size=args.batch_size)
    evaluator.evaluate()
    evaluator.plot_regression(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
