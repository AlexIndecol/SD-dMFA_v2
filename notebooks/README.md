# Notebooks (cookbooks)

These notebooks are designed for **non-programmers** to inspect the model setup, inputs, and results.

Conventions:
- Notebooks auto-detect repo root (they work when launched from repo root or from `notebooks/`).
- Each notebook reads configs from `configs/` and data from `data/exogenous/`.
- Plots use matplotlib defaults (no fixed style/colors).

Suggested usage:
- Open in JupyterLab / Jupyter Notebook.
- For sharing as a lightweight dashboard, consider rendering with `voila` or exporting to HTML/PDF.

Order:
- 00_Quickstart.ipynb
- 01_Config_Explorer.ipynb
- 02_Exogenous_Data_Healthcheck.ipynb
- 03_Stock_and_GAS_Inspector.ipynb
- 04_OD_Trade_Matrix_Viewer.ipynb
- 05_Capacity_Ceiling_Smoothing.ipynb
- 06_Indicators_Viewer.ipynb
