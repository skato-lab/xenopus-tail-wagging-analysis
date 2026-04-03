# Code for analysis of ultra-slow tail wagging in developing Xenopus laevis tadpoles

This repository contains Python scripts used for the analysis of ultra-slow tail wagging in developing *Xenopus laevis* tadpoles.

## Included scripts

- `loess.py`  
  Applies LOWESS/LOESS smoothing to time-series data and outputs detrended signals.

- `fft.py`  
  Performs FFT-based periodicity analysis and outputs peak frequency, period, and related metrics.

- `wavelet_analysis.py`  
  Performs continuous wavelet transform (CWT) analysis and outputs time-resolved period/power information and scalograms.

- `wavelet_band.py`  
  Extracts a representative peak-period band from CWT results and outputs cropped visualizations and summary tables.

## Input data format

The scripts are designed to read CSV files in which:

- the first column is time
- the second column is signal value

Header-containing and headerless CSV files are supported in most cases.

## Requirements

Tested with Python 3.x.

Required packages include:

- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- pywt

Install dependencies with:

```bash
pip install -r requirements.txt
