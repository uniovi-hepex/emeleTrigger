import subprocess, pytest, itertools

CLI_CMDS = [
    "gnn-omtf-data --help",
    "gnn-omtf-train --help",
    "gnn-omtf-opt --help",
    "gnn-omtf-compress --help",
    "gnn-omtf-deploy --help",
    "gnn-omtf-batch --help",
    "gnn-omtf-viz --help",
]

@pytest.mark.parametrize("cmd", CLI_CMDS)
def test_help(cmd):
    assert subprocess.run(cmd.split(), capture_output=True).returncode == 0
