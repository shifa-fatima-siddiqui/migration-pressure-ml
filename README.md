# Migration Pressure Score Prediction

## Overview
This project estimates district-level migration risk in India using machine learning by integrating climate, agricultural, poverty, and demographic data.

## Problem Statement
There is no direct dataset available to measure migration pressure. This project constructs a proxy index to estimate migration risk across districts.

## Dataset Sources
- Census 2011 (demographics)
- IMD Rainfall Data (climate)
- Agriculture Data (yield and production)
- MPI / NFHS (poverty indicators)

## Features Engineered
- Rainfall variability (CV)
- Drought and flood frequency
- Yield and production variability
- Poverty indicators

## Model
- Linear Regression (baseline)
- Random Forest Regressor (final model)
- Achieved R² ≈ 0.90

## Output
- Migration Pressure Score (0–1)
- District-level ranking
- Feature importance analysis

## Dashboard
Interactive Streamlit dashboard for:
- Score visualization
- Feature importance
- District comparisons

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Streamlit

## How to Run
```bash
pip install -r requirements.txt
streamlit run dashboard.py
