"""Pytest bootstrap: put the repo root on sys.path so the flat package
layout (config/, core/, analysis/, ...) imports cleanly from anywhere."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))
