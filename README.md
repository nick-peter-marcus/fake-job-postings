# fake-job-postings
Predicting Fake Job Postings - Data Visualization, TF-IDF, XGBoost, SVC

## Introduction
Looking for a job recently, I was shocked to see that there are so many fake job listings posted on various platforms. Sometimes it quickly became clear that a particular job simply did not exist. The 'recruiters' rather tried to lure you into signing up for additional services like professional teachings, while others demanded creating accounts on websites irrelevant to a job, such as survey platforms.

This is of course incredibly frustrating and infuriating. Coming across this [kaggle dataset](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) inspired me to work out a model that could (hopefully) detect fake jobs.


## Project Scope
Hence, in this project, I will analyze a kaggle dataset containing almost 18,000 job postings, their description, as well as further details such as title, pay rates, benefits, etc. Most importantly, these job postings are labeled as fraudulent / real. This projects consists of 3 steps:
1. Data Cleaning and Recoding: The data will first be rigorously explored, cleaned and relevant features will be engineered (e.g. one-hot-encoding)
2. Exploratory Data Analysis and Data Visualization: Explore distributions and differences of real vs. fake jobs visually.
3. Machine Learning Approaches: two sets of features will be incorporated into a model predicting fake job postings: numeric features, and the TF-IDF vectorized written job description. For both models, cross-validation proposed suitable algorithms for the two analyses: XGBoost for the numeric features, and a Support Vector Classifier for the TF-IDF vectors.


## Key Findings
For the model including numeric features, XGBoost yielded adequate performance with a F1-Score of 0.73 (Precision: 0.84, Recall: 0.64). In terms of feature importance, the length of the written text passages (particularly the company profile), as well as the presence of a company logo contributed substantially to the prediction of fake jobs.

The LinearSVC model - with the TF-IDF vectorized written job posting for features - performed well showing a F1-Score of 0.81 (Precision: 0.99, Recall: 0.68). Investigating the model's coefficients, terms including certain company names, monetary vocabulary and words describing entry-level positions were associated more with fake jobs. On the other hand, positive terms such as 'fun', 'comfortable', and 'passionate' as well as words describing the working environment ('agency', 'clients', 'team', and 'startup') were more likely drawing the classification towards real job postings.