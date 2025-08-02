# redness_tracker

Redness Tracker is a Tkinter application for analysing redness waves in video
sequences.  The project demonstrates an MVC architecture with a background video
processing thread and a collection of small image processing algorithms.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
python -m redness_tracker.app
```

## Testing

```bash
pytest --cov=redness_tracker
mypy --strict redness_tracker
```

## Packaging

A PyInstaller spec file is provided:

```bash
pyinstaller build.spec
```
