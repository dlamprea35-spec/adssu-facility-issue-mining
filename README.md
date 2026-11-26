# Classroom Facility Issue Dashboard — ADSSU

**Classroom Facility Issue Pattern Mining and Maintenance Prioritization Using Text Classification and Association Rule Mining with an Interactive Decision Support Dashboard for Agusan del Sur State University**

Capstone Project 1 — Bachelor of Science in Information Systems (BSIS)

> ⚠️ **Sample dataset for prototype purposes.** The file `data/classroom_issues.csv` is simulated data used to test the system.
> It is **not** an actual ADSSU record, and results from it must not be reported as real findings.
> Replace it with real, authorized records (same columns) before drawing conclusions.

## Problem
Classroom facility problems (electrical, aircon, furniture, AV/IT, plumbing, structural) are reported in free text.
Without analysis, it is hard to know which problems are repeating, which rooms need attention first, and where maintenance resources should go.

## Intended users
Campus maintenance / physical plant office, college deans and facility coordinators, and administrators who plan repairs and budgets.

## Data mining techniques
| Technique | Purpose |
|---|---|
| Text classification (TF-IDF + Logistic Regression / Linear SVM / Naive Bayes) | Automatically categorize new issue reports from their description |
| Association rule mining (Apriori) | Find problems that occur together in the same room, so they can be repaired in one visit |
| Weighted priority scoring | Rank rooms for maintenance and recommend actions |

Evaluation: accuracy, macro precision/recall/F1, confusion matrix, k-fold cross-validation (classification); support, confidence and lift (association rules).

## Dashboard modules
1. Data Exploration (overview, statistics, missing values, charts, drill-down)
2. Text Classification (model evaluation + try-it box)
3. Association Rules (adjustable support / confidence / lift, plain-language interpretation)
4. Decision Support (priority rooms, recommended actions)
5. Report (downloadable summary and CSV files)

## Dataset columns (`data/classroom_issues.csv`)
| Column | Description |
|---|---|
| report_id | Unique report ID |
| timestamp | Date and time reported (`YYYY-MM-DD HH:MM:SS`) |
| campus_location | Campus |
| building_name | College / building |
| room_number | Room code |
| reporter_role | Student or Faculty (no personal names are stored) |
| issue_description | Free-text description of the problem |
| category | Electrical, HVAC/Ventilation, Furniture & Fixtures, IT & Audio-Visual, Plumbing, Structural/Civil |
| urgency_level | Low, Medium, High, Critical |
| resolution_status | Resolved, In Progress, Pending |
| days_to_resolve | Days until fixed (blank while not yet resolved) |

You can also upload your own CSV in the app sidebar if it uses the same columns.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose this repository, branch `main`, main file `app.py`, then **Deploy**.

## Project files
```
app.py            Streamlit dashboard
mining.py         Data preparation, classification, Apriori, prioritization
requirements.txt  Python packages
data/classroom_issues.csv
```

## Ethics and data privacy
Reports contain no names or personal identifiers. Real data must be collected with permission from the ADSSU office concerned and used only for facility planning.
Model outputs are decision-support evidence and do not replace on-site inspection.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, GitHub.
