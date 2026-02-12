# 🏏 IPL Win Probability Predictor

A Machine Learning project that predicts the probability of a team winning an IPL match in real-time during the second innings.

This project uses Logistic Regression, advanced feature engineering, and Streamlit for interactive deployment.

---

## 🚀 Project Overview

Cricket is a game of uncertainty. This project predicts the probability of the chasing team winning an IPL match at any point during the second innings.

The model updates win probability dynamically based on:

- Runs left
- Balls left
- Wickets remaining
- Current Run Rate (CRR)
- Required Run Rate (RRR)
- Batting Team
- Bowling Team
- Match Venue (City)

---

## 📊 Dataset Used

- matches.csv
- deliveries.csv

Source: IPL historical match data

The datasets were merged and processed to extract meaningful match-state features.

---

## 🧠 Feature Engineering

Key engineered features:

- Current Score (Cumulative)
- Runs Left
- Balls Left
- Wickets Left
- Target Score
- Current Run Rate (CRR)
- Required Run Rate (RRR)

Filtering Conditions:

- Only second innings
- No Duckworth–Lewis matches
- Valid chase scenarios

---

## 🤖 Model Used

### Logistic Regression

Why Logistic Regression?

- Suitable for binary classification
- Efficient and interpretable
- Provides probability outputs
- Strong baseline model for sports analytics

---

## 🏗️ Machine Learning Pipeline

- ColumnTransformer for categorical encoding
- OneHotEncoder for teams and city
- Logistic Regression classifier
- Pipeline integration
- Model serialization using Pickle (`pipe.pkl`)

---

## 📈 Model Performance

- Trained on historical IPL chase data
- Evaluated using train-test split
- Achieved competitive classification accuracy
- Outputs win probability using `predict_proba()`

---

## 💻 Streamlit Web App

An interactive web application allows users to:

- Select Batting Team
- Select Bowling Team
- Choose Venue
- Enter:
  - Current Score
  - Wickets Out
  - Overs Completed
  - Target

The app calculates:

- Runs Left
- Balls Left
- Current Run Rate
- Required Run Rate

Then predicts:

- Probability of Batting Team Winning
- Probability of Bowling Team Winning

---

## ▶️ How to Run the Project

### Step 1: Clone Repository

```bash
git clone https://github.com/Keerthi-vasan06/IPL-Win-Probability-Predictor.git
cd IPL-Win-Probability-Predictor

