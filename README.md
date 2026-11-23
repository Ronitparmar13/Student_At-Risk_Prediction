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
| Gmail API | Email communication workflow |

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

See [Student email configuration](#student-email-configuration) to configure Gmail API OAuth credentials in Streamlit secrets.

### Run the Application

```bash
streamlit run app.py
```

*Note: Update the entry-point filename and repository URL to match your actual project files.*

## Student email configuration
The app sends email through the Gmail API over HTTPS, avoiding outbound SMTP
port restrictions on Streamlit Community Cloud. Enable the Gmail API in a
Google Cloud project, configure OAuth consent for the Gmail
`https://www.googleapis.com/auth/gmail.send` scope, and create an OAuth refresh
token for the sending account. Google's
[web server OAuth guide](https://developers.google.com/identity/protocols/oauth2/web-server)
and Gmail [send messages guide](https://developers.google.com/gmail/api/guides/sending)
describe the setup and API.

To obtain the refresh token, create a Web application OAuth client with
`https://developers.google.com/oauthplayground` as an authorized redirect URI.
In [OAuth 2.0 Playground](https://developers.google.com/oauthplayground/),
enable **Use your own OAuth credentials** and enter that client's ID and
secret. Request the Gmail send scope and complete the authorization and token
exchange steps. Keep the refresh token private. If the OAuth consent screen is
in testing mode, add the sending account as a test user; Google may expire
testing refresh tokens after seven days.

Add these values to `.streamlit/secrets.toml` or the deployment's secrets:
```toml
GMAIL_CLIENT_ID = "your-oauth-client-id"
GMAIL_CLIENT_SECRET = "your-oauth-client-secret"
GMAIL_REFRESH_TOKEN = "your-oauth-refresh-token"
EMAIL_FROM = "your-account@gmail.com"
```
Never commit these credentials or share them in chat. `EMAIL_FROM` must be the
Gmail account authorized by the refresh token or one of its configured aliases.

## Tests
```bash
python -m pytest tests/ -v
```

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
