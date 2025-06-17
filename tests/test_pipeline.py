
import subprocess, json, torch, shutil
from pathlib import Path
import pytest

ROOT_STUB = Path(__file__).parent / "smoke_stub.root"
pytestmark = pytest.mark.skipif(
    not ROOT_STUB.exists(), reason="Needs tiny ROOT stub"
)

def run(cmd: str):
    print(">", cmd)
    subprocess.run(cmd, shell=True, check=True)

# --- use a fixed directory that is **not** auto-deleted -----------
DEBUG_DIR = Path(__file__).parent / "debug_smoke"
DEBUG_DIR.mkdir(exist_ok=True)

def test_full_smoke():
    tmp_path = DEBUG_DIR            
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir()

    graphs_pt = tmp_path / "graphs_reg.pt"
    print(f"[DEBUG] graphs.pt → {graphs_pt}")

    run(
        f"gnn-omtf-data convert "
        f"--root-dir {ROOT_STUB} "
        f"--config configs/dataset_regression.yml "
        #f"--max-events 100"
        f"--output {graphs_pt}"
    )
    assert graphs_pt.exists() and graphs_pt.stat().st_size > 0

    # -------------- quick train -----------------------------------
    run(
        f"gnn-omtf-train "
        f"--graphs {graphs_pt} "
        f"--model gat "
        f"--model-args '{{\"hidden_dim\":16}}' "
        f"--epochs 1 "
        f"--batch-size 8 "
        f"--device cpu "
        f"--out-dir {tmp_path/'run'}"
    )
    ckpt = next((tmp_path / "run").rglob("best.ckpt"))
    assert ckpt.exists()

    pruned_dir = tmp_path / "run_compress" / "pruned"
    int8_dir   = tmp_path / "run_compress" / "int8"

    # Run pruning and output to a known folder
    run(f"gnn-omtf-compress prune --ckpt {ckpt} --amount 0.1 --out-dir {pruned_dir}")
    assert (pruned_dir / "model.pt").exists()

    # Dynamic int8 from that folder's model.pt
    run(f"gnn-omtf-compress dyn-int8 --ckpt {pruned_dir / 'model.pt'} --out-dir {int8_dir}")
    assert (int8_dir / "model.pt").exists()


    # --- evaluation (baseline vs compressed) --------------
    run(
        f"gnn-omtf-compress eval "
        f"--baseline   {ckpt} "
        f"--compressed {int8_dir / 'model.pt'} "
        f"--dataset    {graphs_pt} "
        f"--out        {tmp_path/'report.json'}"
    )
    rep = json.loads(Path(tmp_path/"report.json").read_text())
    assert "mse_baseline" in rep and "mse_compressed" in rep
    # --- VIX export --------------------------------------

    run(
        f"gnn-omtf-viz features {graphs_pt} "
        f"--out-dir {tmp_path/'viz_features'}"
    )
    run(
        f"gnn-omtf-viz regression {ckpt} {graphs_pt} "
        f"--out-dir {tmp_path/'viz_reg'}"
    )
    
    run(
        f"gnn-omtf-viz losses {tmp_path/'run'}"
    )
    assert (tmp_path/'run'/'loss_curve.png').exists()
