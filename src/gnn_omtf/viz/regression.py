"""Plot histograms & scatter WITHOUT calling plt.show() (CI-safe)."""
from pathlib import Path
import matplotlib.pyplot as plt

def plot_regression(pred, truth, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # histogram
    plt.figure()
    plt.hist(truth, bins=100, alpha=0.5, label="truth")
    plt.hist(pred,  bins=100, alpha=0.5, label="pred")
    plt.legend(); plt.savefig(out_dir / "hist.png"); plt.close()

    # scatter
    plt.figure()
    plt.scatter(truth, pred, s=4, alpha=0.3)
    plt.xlabel("truth"); plt.ylabel("pred")
    plt.savefig(out_dir / "scatter.png"); plt.close()

    # residuals
    plt.figure()
    plt.hist((pred - truth), bins=100, alpha=0.7)
    plt.savefig(out_dir / "residuals.png"); plt.close()
