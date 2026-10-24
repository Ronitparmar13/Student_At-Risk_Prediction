# Student At-Risk Prediction — VS Code Project

A Streamlit application based on the uploaded `Team4_Student_At_Risk.ipynb` notebook. It includes a dashboard, data cleaning, exploratory data analysis, model training/evaluation, model export, and a prediction form.

## Features

- Upload a student CSV file from the sidebar.
- Clean and inspect records, missing values, duplicates, and out-of-range values.
- Explore target distribution, numeric distributions, and categorical group patterns.
- Train Logistic Regression and/or Random Forest.
- Compare accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Download the cleaned CSV, quality report, and trained model.
- Enter a student's feature values for a model estimate.

## Requirements

- Python 3.10 or newer recommended.
- Visual Studio Code and the Python extension.
- A CSV dataset with a binary `At_Risk` column (0 = not at risk, 1 = at risk).

## Run locally in VS Code

Open this folder in VS Code, then open a terminal in the project directory.

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the app in your browser, normally at `http://localhost:8501`.

## Dataset notes

The notebook's expected target is `At_Risk`. Typical columns referenced include `Attendance_Percentage`, `Assignment_Completion_Percentage`, `Previous_Semester_GPA`, `Previous_Assessment_Score`, `Risk_Score`, `Student_ID`, `Gender`, `Stream`, and `Higher_Education_Interest`. The app tolerates extra columns, but model training requires at least one usable input feature and both target classes.

The app follows the notebook's conservative cleaning approach. Numeric values outside expected ranges are changed to missing rather than deleting whole rows. Imputation, scaling, and encoding are fitted inside a pipeline on the training data only.

## Important interpretation warning

The original notebook checks whether `(Risk_Score >= 50)` exactly reproduces `At_Risk`. If so, that is a strong indication that the label may have been generated from the risk score. `Risk_Score` is excluded from model inputs, but perfect metrics still need investigation for label construction, duplicated students, and other leakage. Validate with an independent dataset before any real student-support use.

The app is an educational prototype, not a clinical or final decision system. Predictions should not be used as the sole basis for consequential decisions about students.

## Main files

- `app.py` — complete Streamlit application
- `requirements.txt` — Python dependencies
- `CODE_REVIEW.md` — review of notable issues in the original notebook
- `data/` — optional location for your dataset (not included)

## Common problems

- **`streamlit` is not recognized:** activate the virtual environment and run `python -m streamlit run app.py`.
- **No target column:** ensure the CSV has a column spelled exactly `At_Risk`.
- **Only one target class:** training requires both 0 and 1.
- **Missing values:** the app imputes feature values inside the training pipeline.
- **Large CSV:** training can take time, especially for Random Forest.
