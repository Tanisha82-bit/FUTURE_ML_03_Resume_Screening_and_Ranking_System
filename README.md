# 💼 AI-Powered Resume Screening & Candidate Ranking System

**An Explainable Machine Learning-Based Recruitment Decision Support System**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-orange)](https://scikit-learn.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-yellow)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

## 📌 Overview

The **AI-Powered Resume Screening & Candidate Ranking System** is a Machine Learning-based application designed to streamline the initial recruitment screening process.

The application analyzes candidate resumes against specific job descriptions, extracts relevant technical skills, calculates job compatibility scores, identifies skill gaps, and generates a ranked list of candidates.

Built using **Python, Streamlit, and Scikit-learn**, the system combines rule-based skill extraction with TF-IDF-based text similarity to provide an explainable and structured candidate evaluation process.

It serves as a decision-support tool for recruiters rather than an autonomous hiring system.

## 🎯 Project Objectives

- Automate initial resume screening and candidate evaluation.
- Extract relevant technical skills from unstructured resume text.
- Measure candidate-job compatibility using Machine Learning techniques.
- Rank candidates based on configurable evaluation criteria.
- Identify missing skills and highlight candidate skill gaps.
- Provide interpretable screening results through an interactive dashboard.

## ⚙️ Technical Approach

The application implements a **Hybrid Scoring Mechanism** combining skill-based matching and text similarity.

### 1. Skill Extraction and Matching

A boundary-aware regular expression-based skill extraction engine identifies technical skills from resumes and job descriptions.

Skills are organized into five categories:

- Programming Languages
- Data Analysis
- Machine Learning
- Databases and Development Tools
- Cloud Computing and Big Data

The extracted skills are compared against the required job skills to calculate the Skill Match Score.

**Weight: 70%**

### 2. TF-IDF Vectorization

The Term Frequency-Inverse Document Frequency (TF-IDF) technique converts resume and job description text into numerical feature vectors.

It helps identify the relevance of candidate profiles to job requirements based on textual similarity.

**Weight: 30%**

### 3. Cosine Similarity

Cosine Similarity measures the similarity between the TF-IDF vectors of a candidate's resume and the target job description.

### 4. Hybrid Candidate Scoring

The final candidate score is calculated using a weighted combination of skill matching and textual similarity.

**Final Score Formula:**

\[
S = (0.70 \times S_{\text{skill}}) +
(0.30 \times S_{\text{TF-IDF}})
\]

Where:

- \(S\) = Final Candidate Match Score
- \(S_{\text{skill}}\) = Skill Match Score
- \(S_{\text{TF-IDF}}\) = Cosine Similarity Score

The weights can be adjusted dynamically through the application's ML Tuner.

### 5. Skill Gap Analysis

The system compares extracted candidate skills with the required job skills and identifies missing competencies.

This feature provides useful insights for candidate feedback and professional upskilling.

## ✨ Key Features

### 📊 Executive Dashboard
- Total number of candidates screened.
- Average candidate match score.
- High-match candidate statistics.
- Overview of screening performance.

### 📌 Job Requirement Configuration
- Enter or customize job descriptions.
- Extract relevant skills from job requirements.
- Add, remove, and manage required skills.
- Configure job-specific screening criteria.

### 📝 Individual Resume Screening
- Upload resumes in PDF, DOCX, and TXT formats.
- Option to paste resume text directly.
- Automatic skill extraction.
- Job compatibility score calculation.
- Detailed skill matching and gap analysis.

### 📁 Batch Resume Screening
- Upload multiple resumes simultaneously.
- Import candidate datasets through CSV files.
- Automatically calculate candidate match scores.
- Generate a ranked candidate leaderboard.
- Export screening results to CSV format.

### 📈 Analytics and Visualizations
- Candidate score distribution.
- Top candidate comparisons.
- Missing skill frequency analysis.
- Interactive data visualizations using Matplotlib and Seaborn.

### ⚙️ ML Tuner and Weight Controls
- Adjust skill matching and TF-IDF weights.
- Configure scoring preferences in real time.
- Explore different candidate ranking outcomes.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Application | Streamlit |
| Machine Learning | Scikit-learn |
| Text Processing | TF-IDF Vectorization |
| Similarity Measurement | Cosine Similarity |
| Skill Extraction | Regular Expressions |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Resume Parsing | PyPDF, Python-docx |
| Dataset | Kaggle Resume Dataset |

## 📂 Dataset

**Dataset:** [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

**Author:** Sneha Anbhawal

The dataset contains resume-related information, including textual resume content, HTML-formatted resumes, and associated job category labels.

It is used as the dataset reference for developing and exploring the resume screening pipeline.

## 📁 Project Structure

```text
AI-Resume-Screening/
│
├── app.py
│   └── Streamlit-based web application
│
├── index.html
│   └── Standalone web application prototype
│
├── Resume_Screening.ipynb
│   └── ML pipeline and exploratory data analysis
│
├── requirements.txt
│   └── Project dependencies
│
├── candidate_ranking_results.csv
│   └── Exported candidate screening results
│
└── README.md
    └── Project documentation
```

## 🚀 Installation and Setup

Follow these steps to run the application locally.

### Prerequisites

- Python 3.9 or later
- Git
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/AI-Resume-Screening.git
```

### Step 2: Navigate to the Project Directory

```bash
cd AI-Resume-Screening
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Application

```bash
streamlit run app.py
```

The application will be accessible through your browser at:

```text
http://localhost:8501
```

## 📦 Dependencies

The project uses the following Python libraries:

```text
streamlit>=1.25.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pypdf>=3.0.0
python-docx>=0.8.11
```

## ⚖️ Ethical Considerations and Responsible AI

This project is designed as a **human-in-the-loop recruitment decision-support prototype**.

Key considerations include:

- Focuses on job-related skills and resume content.
- Does not intentionally use protected demographic attributes such as gender, age, or ethnicity in its scoring formula.
- Provides skill-match explanations and missing-skill insights.
- Supports recruiter evaluation rather than replacing human judgment.

**Important:** Text-based screening systems can still introduce unintended biases through resume content, training data, or evaluation criteria. Final recruitment decisions should always involve human review.

## 🔮 Future Enhancements

Potential improvements include:

- Integration of advanced NLP and transformer-based language models.
- Semantic resume-job description matching.
- Improved entity recognition for technical skills.
- Explainable AI techniques for candidate score interpretation.
- Database integration for candidate management.
- Cloud deployment and REST API integration.
- Expanded bias evaluation and fairness testing.

## 🙌 Acknowledgments

- **Future Interns Internship Program** – Task 03: Machine Learning Assignment.
- **Kaggle** – For providing the Resume Dataset.
- **Scikit-learn** – For machine learning and text processing tools.
- **Streamlit** – For interactive application development.

---

**Developed as a Machine Learning project focused on recruitment automation, text analysis, and explainable candidate evaluation.**
