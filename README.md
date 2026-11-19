# 🎓 Student Success & At-Risk Prediction Dashboard

An interactive Machine Learning-powered dashboard that identifies students who may be at academic risk by analyzing attendance and assignment performance.

The application helps educators monitor student progress, identify potential academic challenges, and support timely intervention through data-driven insights.

## 🚀 Project Overview

Student performance can be affected by several factors, including attendance and assignment completion.

This project uses Machine Learning classification models to predict student risk categories and presents the results through an interactive Streamlit dashboard.

It enables users to explore student records, filter students by risk status, analyze performance trends, and initiate personalized student support communication.

## ✨ Key Features

- **Student Risk Prediction:** Identifies students who may be academically at risk.
- **Machine Learning Models:** Supports Logistic Regression and Random Forest.
- **Interactive Student Roster:** Filter students into All Students, At Risk, and Not at Risk categories.
- **Student Search:** Search for individual student records using student identifiers.
- **Attendance Analysis:** Visualizes attendance performance with a 60% reference threshold.
- **Assignment Analysis:** Visualizes assignment performance with a 70% reference threshold.
- **Interactive Dashboard:** Displays student performance insights through visualizations.
- **Student Dataset:** Provides access to student-level information and prediction results.
- **Student Support Workflow:** Helps prepare personalized email communication for students identified as at risk.
- **Email Review and Sending:** Allows the user to review and confirm an email draft before sending it.

## 🧠 Machine Learning Approach

The dashboard supports two classification algorithms:

### 1. Logistic Regression
A supervised classification algorithm used to estimate the probability of a student belonging to an academic risk category.

### 2. Random Forest
An ensemble learning algorithm that combines multiple decision trees to classify student risk based on available features.

Users can select the model and explore the resulting predictions.

## 📊 Dashboard Components

### Student Roster
- View all students.
- Filter at-risk and not-at-risk students.
- Search individual student records.
- Select the Machine Learning model.

### Attendance and Assignment Analytics
- Visualize attendance performance.
- Compare assignment performance.
- Identify students falling below the defined thresholds.
- Explore performance patterns across student records.

### Student Dataset
Displays student-level information, including Student_ID, academic indicators, and prediction results.

### Student Support
Provides a workflow to identify students requiring attention, prepare a formal progress-related email, review the draft, and send it after confirmation.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data manipulation and analysis |
| Scikit-learn | Machine Learning models |
| Logistic Regression | Student risk classification |
| Random Forest | Ensemble classification |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualizations |
| SMTP / Gmail | Email communication workflow |

## ⚙️ Getting Started

### Prerequisites
- Python 3.10 or compatible version
- Git
- VS Code or another Python IDE

### Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Navigate to the Project Folder

```bash
cd YOUR_PROJECT_FOLDER
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Email Credentials

Configure Gmail SMTP credentials using Streamlit secrets or environment variables.

Never expose your Gmail password or app password in source code or commit it to GitHub.

### Run the Application

```bash
streamlit run app.py
```

*Note: Update the entry-point filename and repository URL to match your actual project files.*

## 💡 What I Learned

- Building end-to-end Machine Learning applications.
- Implementing classification models using Scikit-learn.
- Performing data analysis and preprocessing with Pandas.
- Creating interactive dashboards using Streamlit and Plotly.
- Translating model predictions into understandable visual insights.
- Integrating email workflows into a Python application.
- Designing user-friendly tools to support data-driven decisions.

## 🔮 Future Enhancements

- Add more academic performance indicators.
- Evaluate models using precision, recall, F1-score, and confusion matrices.
- Add explainable AI to interpret individual risk predictions.
- Introduce automated academic performance reports.
- Expand student support and intervention tracking.


---

*Developed as a practical Machine Learning and Streamlit project focused on identifying academic risk and supporting student success.*
