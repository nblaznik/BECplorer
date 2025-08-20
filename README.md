<!-- Optional banner -->
<!-- ![BECplorer](docs/banner.png) -->

<div align="center" style="display: flex; align-items: center; justify-content: center;">
  <div style="flex: 1; max-width: 10%; text-align: center;">
    <img width="128" height="128" alt="icon" src="https://github.com/user-attachments/assets/b66db70b-0979-4a14-be07-a72687521d06" />
  </div>
  <div style="flex: 9; max-width: 90%; text-align: left; padding-left: 20px;">
    <h1>BECplorer — BEC Data Visualiser</h1>
    <p>
      A lightweight framework for inspecting and analyzing <code>.fits</code> images from the UU BEC Lab.<br>
      Built with PyQt5. Special features for analysis of cold atoms data obtained through Off-axis holography.
    </p>
  </div>
</div>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick start</a> •
  <a href="#usage">Usage</a>
</p>

---

## Features

- **Fast FITS viewer**: open large image stacks; lazy-loading for speed.
- **Live plotting and analysis**: real-time updates and analysis of the incoming data. Features such as live particle number, temperature and cooling efficiency.
- **Interactive inspection**: pan/zoom, pixel probe, ROI selection, linecuts.
- **Batch tools**: normalization, cropping, mask application.
- **BEC-specific analysis**:
  - OD / phase / ratio images
  - Axial and radial linecuts with Gaussian/TF fits
- **Annotations**: overlays for ROIs, scalebars, fit results; export as PNG/SVG.
- **Comment**: enables commenting and exporting batch comments.
- **Session management**: autosave session state; reopen where you left off.
- **Export**: figures, CSVs of fit parameters, and MP4/GIF for time series.
- **SOAH-features**: fast-Fourier transform analysis of interference patterns, extraction of the entire field. For details and the theory behind it see Blaznik's PhD thesis. 
- **SVD and PCA-features**: principal-component analysis features - using large datasets of background images, a PCA analysis can be performed in order to minimize the noise of the phase images obtained through SOAH. For detailes see Blaznik's PhD thesis, or in the <a href="https://orcid.org/0009-0003-7288-719X">article</a>.

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

# 4) Run
python -m gui.py
