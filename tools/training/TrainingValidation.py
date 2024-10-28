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