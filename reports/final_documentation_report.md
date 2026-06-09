# Mental Health Sentiment Analyzer - Final Documentation Report

## 1. Project Objective

The objective of this task is to build a Natural Language Processing system that analyzes employee mental health survey responses, classifies sentiment, identifies at-risk employee cohorts, and presents insights through a professional Streamlit application.

The system is designed to help HR teams understand workplace wellness patterns from employee survey data.

## 2. Dataset

- Dataset: OSMI Mental Health in Tech Survey
- Source: Kaggle
- Raw records: 1,259
- Cleaned records: 1,250
- Usable free-text comment records for sentiment modeling: 160

The dataset contains employee demographics, workplace mental health support indicators, treatment status, work interference, and optional free-text comments.

## 3. Work Completed

- Loaded and inspected the raw survey dataset.
- Checked missing values, duplicate records, and inconsistent values.
- Cleaned invalid age values.
- Standardized gender values into Male, Female, and Other.
- Filled important missing categorical fields where appropriate.
- Created a cleaned survey dataset.
- Performed exploratory data analysis.
- Generated visuals for age, gender, country, family history, treatment, and work interference.
- Preprocessed free-text employee comments.
- Created sentiment labels using TextBlob polarity because the dataset does not provide manual sentiment labels.
- Converted text comments into TF-IDF numerical features.
- Trained multiple sentiment classification models.
- Compared model performance using Accuracy and Weighted F1 score.
- Selected the best-performing sentiment model.
- Exported the trained NLP pipeline.
- Generated sentiment analysis, at-risk cohort, and HR recommendation reports.
- Developed a professional Streamlit dashboard application.

## 4. Methodology

### Data Cleaning

The raw dataset was cleaned by:

- Removing invalid age values outside the expected employee age range.
- Standardizing gender entries.
- Filling missing `state`, `self_employed`, and `work_interfere` values.
- Keeping usable employee comments for NLP analysis.

### Text Preprocessing

Employee comments were processed by:

- Converting text to lowercase.
- Removing URLs.
- Removing special characters and numbers.
- Normalizing extra whitespace.

### Sentiment Label Creation

The dataset does not include sentiment labels. Therefore, TextBlob polarity was used to generate labels:

- `Positive`: polarity greater than 0.05
- `Negative`: polarity less than -0.05
- `Neutral`: polarity between -0.05 and 0.05

### Feature Extraction

TF-IDF Vectorization was used to convert cleaned comments into numerical features for model training.

### Model Training

The following models were trained:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- Tuned Logistic Regression

The final model was selected using Weighted F1 score, with Accuracy as the secondary metric.

## 5. Model Results

| Model | Accuracy | Weighted F1 |
| --- | ---: | ---: |
| Logistic Regression | 0.425 | 0.4197 |
| Tuned Logistic Regression | 0.425 | 0.4072 |
| Naive Bayes | 0.500 | 0.3745 |
| Linear SVM | 0.375 | 0.3619 |

## 6. Selected Model

The selected model is:

**TF-IDF Vectorizer + Logistic Regression**

It was selected because it achieved the highest Weighted F1 score, meaning it produced the best overall balance across Positive, Negative, and Neutral sentiment classes.

The model is exported as:

`models/sentiment_tfidf_pipeline.pkl`

## 7. Key Findings

- The cleaned dataset contains 1,250 survey responses.
- Only 160 rows contain usable free-text comments, which limits sentiment model accuracy.
- Positive comments are the largest sentiment group.
- A meaningful number of comments are negative and may indicate workplace wellness concerns.
- Employees with family history show higher treatment-seeking behavior.
- Employees reporting work interference are more likely to have mental health concerns.
- Role-based analysis could not be completed because the source dataset does not contain a role/job-title column.

## 8. Streamlit Application

The Streamlit application is implemented in:

`app.py`

The application includes:

- Executive overview
- Sentiment analysis dashboard
- At-risk cohort dashboard
- Model performance dashboard
- New comment sentiment prediction
- Documentation and reports viewer

For new comment prediction, the app displays the trained TF-IDF model output and also applies a polarity/risk-term guard so short high-risk phrases such as stress, burnout, anxiety, or depression are flagged more appropriately for review.

## 9. Achievements

- Completed an end-to-end NLP workflow.
- Built and exported a reusable sentiment classification pipeline.
- Created a professional dashboard for analysis and prediction.
- Prepared complete documentation and supporting reports.
- Organized datasets, models, reports, scripts, notebooks, and visuals for final ZIP submission.

## 10. Limitations

- Sentiment labels are automatically generated using TextBlob, not manually annotated by domain experts.
- The free-text comment dataset is small, which affects model accuracy.
- The system is not a medical, clinical, or employment decision tool.
- Role-based cohort analysis is not available due to missing role/job-title data.

## 11. Final Submission Contents

The final ZIP should include:

- `app.py`
- `requirements.txt`
- `README.md`
- `Data/`
- `models/`
- `reports/`
- `visuals/`
- `Notebooks/`
- `scripts/`

## 12. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```
