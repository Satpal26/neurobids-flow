# cli.py
# Command line interface for eeg2bids-unify
# Usage: uv run python -m neurobids_flow.cli convert --file data.vhdr --bids-root ./output --subject 01 --session 01 --task ssvep

import click
from .core.converter import EEGConverter


@click.group()
def cli():
    """neurobids-flow — Standardize multi-source EEG recordings to BIDS-EEG format."""
    pass


@cli.command()
@click.option("--file", required=True, help="Path to raw EEG file")
@click.option("--bids-root", required=True, help="Path to output BIDS folder")
@click.option("--subject", required=True, help="Subject ID e.g. 01")
@click.option("--session", required=True, help="Session ID e.g. 01")
@click.option("--task", required=True, help="Task name e.g. ssvep")
def convert(file, bids_root, subject, session, task):
    """Convert a single EEG file to BIDS format."""
    converter = EEGConverter()
    bids_path = converter.convert(
        filepath=file,
        bids_root=bids_root,
        subject=subject,
        session=session,
        task=task,
    )
    click.echo(f"Done! BIDS dataset written to: {bids_root}")


if __name__ == "__main__":
    cli()
