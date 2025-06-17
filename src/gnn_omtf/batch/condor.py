from __future__ import annotations

"""
Thin wrapper that renders  (job.sh  +  submit.sub)  from Jinja2 templates
and optionally submits them to HTCondor.

All heavy lifting (chunking, CLI parsing) happens in *cli.py*; this module
only deals with file I/O + `condor_submit`.
"""


import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from jinja2 import Environment, FileSystemLoader

from . import templates as _tmpl_dir

log = logging.getLogger(__name__)


_env = Environment(loader=FileSystemLoader(str(_tmpl_dir)))


def _render(template: str, ctx: Mapping[str, object]) -> str:
    return _env.get_template(template).render(**ctx)


def render_and_submit(jobs: Iterable[dict], *, dry_run: bool = False) -> None:
    """
    Parameters
    ----------
    jobs :
        Iterable of dictionaries, each **at least** containing keys:

        * ``idx``       – int  (job number)
        * ``cmd``       – str  (e.g. 'gnn-omtf-data')
        * ``args``      – str  (CLI arguments)
        * ``queue``     – str  (HTCondor JobFlavour)
    dry_run :
        If *True* just dump the submit file on stdout and stop.
    """
    work = Path(tempfile.mkdtemp(prefix="gnn-omtf_"))
    log.info("batch work-dir → %s", work)
    logs = work / "logs"
    logs.mkdir()

    # 1. render one job wrapper **per** entry
    for job in jobs:
        js_path = work / f"job_{job['idx']:04d}.sh"
        job["job_script"] = js_path
        job["logs"] = logs
        js_path.write_text(
            _render(
                "job.sh.j2",
                {
                    "env": "$HOME/pyenv/bin/activate",  # customise here
                    **job,
                },
            )
        )
        js_path.chmod(0o755)

    # 2. single submit.sub referencing the wrapper scripts
    submit = work / "submit.sub"
    submit.write_text(
        _render(
            "submit.sub.j2",
            {
                "queue": jobs.__iter__().__next__()["queue"],  # use first entry
                "logs": logs,
            },
        )
    )

    if dry_run:
        print(submit.read_text())
        log.info("Dry-run → no submission")
        return

    # 3. fire away
    subprocess.run(["condor_submit", str(submit)], check=True)
    log.info("Submitted %d jobs", len(list(jobs)))
