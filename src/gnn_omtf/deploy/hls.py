import typer

hls_cli = typer.Typer(help="HLS utilities (build, synthesize, benchmark)")

@hls_cli.command("dummy")
def dummy():
    """Placeholder command for CLI discovery."""
    typer.echo("✅ HLS CLI is reachable")
