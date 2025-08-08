# Mental Health in Tech — Streamlit Dashboard

## Project summary
This project explores the OSMI Mental Health in Tech survey data and provides an interactive Streamlit dashboard to:
- Explore the dataset with simple visualizations (age distribution, treatment by gender, family history).
- Predict treatment-seeking likelihood (classification).
- Estimate respondent age from workplace/attitude features (regression).
- Discover mental-health-related personas using clustering (KMeans) with a PCA scatter plot.

Primary artifacts in this repo:
- `app.py` — Streamlit app (dashboard).
- `survey.csv` — Input dataset expected by the app.
- `EDA.ipynb`, `classification_model.ipynb`, `clustering.ipynb`, `regression_model.ipynb` — Notebooks with supporting analysis and modeling experiments.

## Setup instructions
Prerequisites:
- Python 3.8+ (3.10–3.12 recommended; 3.13 also works).
- pip
- Windows PowerShell (commands below use PowerShell syntax).

1) Clone or open this project folder.

2) Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

3) Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# If you don’t want a requirements file, install directly:
# python -m pip install streamlit pandas numpy seaborn matplotlib scikit-learn xgboost
```

4) Run the Streamlit app:
```powershell
# Recommended (works even if Scripts path isn’t on PATH):
python -m streamlit run app.py

# Or, if streamlit is on PATH:
# streamlit run app.py
```

Once started, open the app in your browser at:
- Local: http://localhost:8501
- Network (auto-shown in terminal): http://<your-ip>:8501

Troubleshooting:
- If you see “streamlit is not recognized…”, use `python -m streamlit run app.py`.
- Pandas “SettingWithCopy”/“FutureWarning” messages are benign for app use; they come from chained assignments during cleaning.

## Feature description
The dashboard (left sidebar navigation) includes:
- Introduction: Project overview and guidance.
- About Data: Dataset snapshot and summary (intended to show a preview with a toggle to expand).
- Exploratory Data Analysis:
  - Age distribution (downloadable PNG).
  - Treatment seeking by gender (downloadable PNG).
  - Family history pie chart (downloadable PNG).
- Treatment Prediction: Logistic Regression trained on one-hot-encoded features to predict likelihood of seeking treatment.
- Age Prediction: Random Forest Regressor to estimate respondent age from workplace and attitude features.
- Persona Clustering:
  - KMeans (k=3) on selected features, scaled with StandardScaler.
  - PCA 2D visualization of clusters.
  - Personas mapped to cluster labels (e.g., Open Advocates, Under-Supported Professionals, Silent Sufferers) with downloadable visualization.

Notes:
- The app expects `survey.csv` in the project root.
- The notebooks provide deeper EDA and modeling experiments (classification, clustering, regression).

## Deployed app link
This project isn’t published to a public URL yet. For local development use:
- http://localhost:8501

To deploy (Streamlit Community Cloud):
1) Push this project to a public GitHub repository (include `app.py`, `survey.csv`, and `requirements.txt`).
2) Go to https://share.streamlit.io/, sign in, and create a new app.
3) Select your repo, branch, and `app.py` as the entrypoint.
4) Click Deploy. You’ll receive a URL like `https://<your-app-name>.streamlit.app`.
5) Update this README section with that URL once deployed.

## Data
- Source file: `survey.csv` (place in project root). Ensure column names match those expected by `app.py` (e.g., Age, Gender, treatment, family_history, etc.).

## License
Add your preferred license here (e.g., MIT, Apache 2.0).
