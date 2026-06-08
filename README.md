# Mental Health Sentiment Analyzer

## Project Overview

This project follows the internship task PDF for building an NLP workflow that analyzes employee mental health survey responses, classifies comment sentiment, identifies at-risk cohorts, and prepares inputs for an Employee Wellness Dashboard.

The dashboard itself has not been created yet, as requested. All work up to the dashboard-building step is prepared.

## Dataset

Source: https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

The dataset contains employee mental health survey responses from technology professionals, including demographics, workplace support factors, treatment status, and optional free-text comments.

## Current Progress Against PDF

### Completed

- Downloaded and inspected the dataset.
- Cleaned age, gender, state, self-employment, and work-interference fields.
- Created a cleaned dataset.
- Performed missing-value, duplicate, demographic, and workplace-factor analysis.
- Created EDA visualizations.
- Preprocessed free-text comments.
- Generated word cloud and word-frequency style NLP outputs.
- Created train/test splits.
- Converted text comments into TF-IDF features.
- Trained Naive Bayes, Logistic Regression, and Linear SVM sentiment classifiers.
- Compared models using accuracy and weighted F1 score.
- Generated classification report and confusion matrix.
- Performed basic Logistic Regression tuning.
- Selected and exported the best sentiment pipeline.
- Analyzed sentiment distribution across employee groups.
- Identified at-risk cohorts using treatment, family history, work interference, age group, country, and negative sentiment.
- Prepared HR insights and recommendation reports.

### Not Done Yet

- Employee Wellness Dashboard.

This is the next step in the PDF after the current completed work.

## Important Note About Sentiment Labels

The Kaggle survey dataset does not include manually labeled sentiment classes. To complete the required sentiment workflow, this project uses TextBlob polarity to create sentiment labels from employee comments, then trains TF-IDF based classifiers on those generated labels.

Because there are only 161 usable comment rows, the sentiment model is useful as a project workflow artifact but should not be treated as a production HR or clinical system.

## Latest Model Result

Best text sentiment model: Logistic Regression

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.425 | 0.4197 |
| Tuned Logistic Regression | 0.425 | 0.4072 |
| Naive Bayes | 0.500 | 0.3745 |
| Linear SVM | 0.375 | 0.3619 |

Earlier notebook work also trained a survey-feature model to predict `treatment`, where Logistic Regression reached 83.2% accuracy. That is useful for mental-health risk analysis, but it is not the same as the PDF's requested text sentiment classifier.

## Project Structure

```text
Mental_Health_Sentiment_Analyzer/
|-- Data/
|   |-- survey.csv
|   |-- cleaned_survey.csv
|   `-- processed_comments_sentiment.csv
|-- Notebooks/
|   |-- day1_eda.ipynb
|   |-- day2.ipynb
|   `-- day2_sentiment_modeling.ipynb
|-- models/
|   |-- mental_health_model.pkl
|   `-- sentiment_tfidf_pipeline.pkl
|-- reports/
|   |-- sentiment_analysis_report.md
|   |-- at_risk_cohort_report.md
|   |-- hr_recommendations.md
|   |-- findings_summary_before_dashboard.md
|   |-- sentiment_model_results.csv
|   |-- sentiment_distribution.csv
|   |-- sentiment_by_gender.csv
|   |-- sentiment_by_country_top10.csv
|   |-- treatment_by_family_history.csv
|   |-- treatment_by_work_interference.csv
|   `-- treatment_by_age_group.csv
|-- scripts/
|   `-- complete_until_dashboard.py
|-- visuals/
|   |-- age_distribution.png
|   |-- country_distribution.png
|   |-- family_history_distribution.png
|   |-- gender_distribution.png
|   |-- mental_health_wordcloud.png
|   |-- treatment_distribution.png
|   |-- work_interfere_distribution.png
|   |-- sentiment_distribution.png
|   |-- sentiment_model_comparison.png
|   `-- sentiment_confusion_matrix.png
|-- README.md
|-- requirements.txt
`-- .gitignore
```

## Reproduce Outputs

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pre-dashboard pipeline:

```bash
python scripts/complete_until_dashboard.py
```

This regenerates the processed comment dataset, EDA visuals, sentiment model, model metrics, and pre-dashboard reports.

## Next Step

Build the Employee Wellness Dashboard using the prepared files in `Data/`, `reports/`, `visuals/`, and `models/`.
