# Code review: `Team4_Student_At_Risk.ipynb`

This review is based on the uploaded notebook's source code. I did not have the original CSV dataset, so I could not verify dataset-specific counts, column values, or reproduce the reported model scores.

## Key findings

### 1. Google Colab-only code prevents direct local execution
The notebook uses `from google.colab import files`, `files.upload()`, and `files.download()`. These are Colab-specific and will fail in a normal VS Code Python environment. Replace them with a local file picker, command-line path, or web UI upload. The included Streamlit app uses a CSV uploader and download buttons.

### 2. Potential target leakage needs investigation
The notebook explicitly checks whether `(Risk_Score >= 50)` matches `At_Risk` for every row and reports that this was true in a previously inspected CSV. The model code excludes `Risk_Score`, which is correct if it directly defines the target. Confirm the label-generation process and dataset provenance before describing the task as prediction of future risk.

### 3. Model-selection branches can cause downstream KeyErrors
The training cell allows Logistic Regression, Random Forest, or Both. Later cells access `trained_models["Random Forest"]` directly (including the Random Forest validation and save cells). If only Logistic Regression was selected, those later cells can raise `KeyError`. The UI checks which models were actually trained.

### 4. Some later cells assume columns always exist
The cell calculating average feature values groups by `At_Risk` and indexes a fixed list of four academic columns without checking that all four exist. A CSV with a different schema can fail. The app checks available columns before showing group charts.

### 5. Export summary assumes `At_Risk` exists
The EDA export builds `summary["target_counts"]` from `df_clean["At_Risk"]` even though some earlier cells check whether the target exists conditionally. If `At_Risk` is missing, export will fail. The app reports a clear missing-target message.

### 6. Perfect Random Forest metrics are not proof of generalization
The notebook reports Random Forest precision, recall, and F1-score of 1.00, and zero false positives/negatives. Those results should trigger checks for label construction, duplicates or overlapping entities, and independent validation. The exact `Risk_Score` rule is one plausible explanation to investigate, not proof of the cause.

### 7. Split-before-preprocessing is implemented appropriately in the model pipeline
The notebook's later model pipeline splits data before fitting preprocessing and builds a combined preprocessing/model pipeline. This helps avoid preprocessing leakage. The earlier standalone preprocessing cell is redundant for training because the model-training cell builds a fresh pipeline.

### 8. The target labels must be valid binary values
The notebook checks that `At_Risk` contains both 0 and 1 and uses stratification. This is good, but it can fail for tiny datasets or if one class has fewer than two records. The app surfaces these cases as readable errors.

### 9. Identifier handling deserves care
The notebook excludes `Student_ID` and `Gender` from model inputs. This is preserved in the app. If a student can have multiple rows, consider grouping or splitting by student identity so the same student's records do not appear in both training and test sets.

### 10. Scope mismatch in notebook description
The notebook's opening description says the notebook completes cleaning and EDA only, but later cells also train, compare, evaluate, and save models. Update the project documentation so its scope is clear.

## App implementation choices

- Streamlit was selected because it works well for data-science projects and is straightforward to run in VS Code.
- The UI includes cleaning, exploratory analysis, model evaluation, and a manual prediction form.
- `Risk_Score`, `Student_ID`, `Gender`, and the target are excluded from training features, matching the notebook's model-stage choices.
- Model preprocessing is fit only on the training split.
- The app is a prototype. Review model quality and fairness with suitable independent data and human oversight.
