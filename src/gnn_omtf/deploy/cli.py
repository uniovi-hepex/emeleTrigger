from __future__ import annotations
import typer

from .onnx import onnx_cli
from .hls  import hls_cli        # <-- new

app = typer.Typer(help="Deployment helpers (ONNX export, hls synthesis)")
app.add_typer(onnx_cli, name="onnx")
app.add_typer(hls_cli,  name="hls")     # <-- new


def _main() -> None:   # console-script entry-point
    app()


if __name__ == "__main__":
    _main()
