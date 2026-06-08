# Mental Health Sentiment Analyzer

## Project Overview

Mental Health Sentiment Analyzer is an NLP-based project that analyzes employee mental health survey responses and identifies signs of stress, anxiety, burnout, and workplace well-being concerns.

The system helps HR teams understand employee sentiment, identify at-risk cohorts, and generate actionable workplace wellness insights.

---

## Objectives

- Analyze employee mental health survey data
- Perform text preprocessing and sentiment analysis
- Explore demographic patterns affecting mental health
- Build and evaluate sentiment classification models
- Generate HR-focused wellness insights
- Create visual dashboards for decision making

---

## Dataset

Dataset Source:

https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

The dataset contains responses from technology professionals regarding workplace mental health, treatment, support systems, and demographics.

---

## Project Workflow

### Day 1: Data Preparation & Exploratory Analysis

- Data loading and inspection
- Missing value analysis
- Duplicate detection
- Data cleaning
- Demographic analysis
- Text preprocessing
- Tokenization
- Stopword removal
- Text normalization
- Word frequency analysis
- Word cloud generation
- Exploratory Data Analysis (EDA)

### Day 2: Sentiment Classification & Dashboard

- TF-IDF Vectorization
- Count Vectorization
- Naive Bayes Classification
- Logistic Regression
- Support Vector Machine (SVM)
- Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- At-risk cohort identification
- Wellness dashboard development
- HR recommendations generation

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- WordCloud
- Jupyter Notebook

---

## Project Structure

```text
Mental_Health_Sentiment_Analyzer/
│
├── Data/
│   └── survey.csv
│
├── Notebooks/
│   └── day1_eda.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore