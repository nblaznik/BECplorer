<!-- Optional banner -->
<!-- ![BECplorer](docs/banner.png) -->

<h1 align="center">BECplorer — BEC Data Visualiser</h1>
<p align="center">
A lightweight framework for inspecting and analyzing <code>.fits</code> images from the UU BEC Lab.<br>
Built with PyQt5. Cross-platform. Extensible.
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#usage">Usage</a> •
  <a href="#plugins--extensibility">Plugins</a> •
  <a href="#roadmap">Roadmap</a> •
  <a href="#citing">Citing</a>
</p>

---

## Features

- **Fast FITS viewer**: open large image stacks; lazy-loading for speed.
- **Interactive inspection**: pan/zoom, pixel probe, ROI selection, linecuts.
- **Batch tools**: normalization, cropping, mask application.
- **BEC-specific analysis**:
  - OD / phase / ratio images
  - Axial and radial linecuts with Gaussian/TF fits
- **Annotations**: overlays for ROIs, scalebars, fit results; export as PNG/SVG.
- **Comment**: enables commenting and exporting batch comments.
- **Session management**: autosave session state; reopen where you left off.
- **Export**: figures, CSVs of fit parameters, and MP4/GIF for time series.

> Screenshots  
> <img width="365" height="192" alt="BEC_Viewer (4)" src="https://github.com/user-attachments/assets/2807d665-2cd4-4ddf-9035-fb0d464c232a" />
> <img width="364" height="192" alt="BEC_Viewer (1)" src="https://github.com/user-attachments/assets/6dc246c3-8f71-40b4-a92d-8c43aa0caa49" />
> <img width="364" height="197" alt="BEC_Viewer" src="https://github.com/user-attachments/assets/4eed33ef-a853-43cf-b532-727df438d98a" />
> <img width="364" height="192" alt="BEC_Viewer (2)" src="https://github.com/user-attachments/assets/2a2854e1-10f9-4223-a065-0aa7d6440942" />

---

## Requirements

- Python ≥ 3.9
- PyQt5, numpy, scipy, astropy, matplotlib (see `requirements.txt`)

---

## Quick start

```bash
# 1) Clone
git clone https://github.com/<org-or-user>/<repo>.git
cd <repo>

# 2) Create env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install
pip install -U pip
pip install -r requirements.txt
# or: pip install -e .   # if you provide a pyproject.toml/setup.cfg

# 4) Run
python -m becplorer
# or: becplorer           # if installed as a console script
