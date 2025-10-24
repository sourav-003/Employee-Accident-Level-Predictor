# Understanding Employee Accidents in Plant

**Objective:**  
Analyze and predict the severity of workplace accidents using both structured and unstructured data (incident descriptions).  
This project explores how machine learning and NLP can uncover root causes and help organizations prevent future incidents.

---

##  Project Overview

This case study focuses on **employee accident data from an industrial plant**, combining **Exploratory Data Analysis (EDA)**, **Natural Language Processing (NLP)**, and **Machine Learning** to:

- Identify patterns behind different accident levels  
- Understand the impact of gender, sector, and employee type  
- Analyze textual accident descriptions for sentiment and keywords  
- Build predictive models for `Accident Level` and `Potential Accident Level`  
- Suggest data-driven strategies for workplace safety improvement  

---

##  Tech Stack & Tools

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Holoviews, WordCloud |
| **NLP** | NLTK, Scikit-learn (TF-IDF), VADER Sentiment |
| **Machine Learning** | RandomForest, XGBoost, LightGBM, Logistic Regression, Decision Tree, Bagging |
| **Data Balancing** | SMOTE, RandomOverSampler |
| **Model Persistence** | Joblib |
| **Deployment (Prototype)** | Gradio |
| **Statistical Testing** | Proportion Z-Test (statsmodels) |

---

##  Workflow Summary

### 1. Data Understanding & Cleaning
- Loaded and inspected the dataset  
- Removed redundant columns (`Unnamed: 0`) and handled missing values  
- Encoded categorical variables for modeling  

### 2. Exploratory Data Analysis (EDA)
- Visualized accident distribution by gender, sector, and job type  
- Generated percentage breakdowns for `Accident Level` and `Potential Accident Level`  
- Created word clouds from processed accident descriptions  

### 3. Text Preprocessing & Sentiment Analysis
- Tokenization, stopword removal, and lemmatization using **NLTK**  
- Performed **sentiment analysis** using **VADER**  
- Extracted **TF-IDF** features (`max_features=100`) to represent textual descriptions numerically  

### 4. Hypothesis Testing
- Conducted a **Proportion Z-Test** to check if accident proportions differ significantly between genders  

### 5. Feature Engineering
- Combined categorical, numeric, sentiment, and TF-IDF features  
- Addressed **class imbalance** using SMOTE / oversampling techniques  

### 6. Model Building
- Trained and evaluated multiple classifiers:
  - RandomForest  
  - XGBoost  
  - LightGBM  
  - Logistic Regression  
  - Decision Tree  
  - Bagging Classifier  

- Evaluated using:
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-score)

### 7. Deployment
- Saved the trained models using **Joblib**  
- Built a **Gradio app prototype** for interactive prediction of accident severity  

---

##  Key Insights

- Textual descriptions such as *“slip,” “fall,”* and *“machine”* strongly correlate with higher accident severity.  
- Certain employee sectors show greater exposure to severe incidents.  
- Gender-wise differences in accident involvement were found statistically significant.  
- Ensemble models (XGBoost, LightGBM) delivered the best macro F1-scores.  

---

##  Learnings

- Integrated **NLP with structured data** for holistic accident analysis.  
- Understood the importance of **class imbalance handling** (SMOTE) in multi-class prediction.  
- Demonstrated the use of **statistical testing** (Z-Test) to validate assumptions.  
- Built an end-to-end **data-to-deployment pipeline** using modern Python tools.

---

##  Future Enhancements

- Implement **cross-validation** and **hyperparameter tuning** for model optimization.  
- Replace TF-IDF with **transformer embeddings (BERT / Sentence-Transformers)** for richer text understanding.   
- Develop a **Streamlit dashboard** for real-time safety analytics.  
- Integrate with Power BI for visualization and management insights.

---

##  Results Summary

| Model | Accuracy | Macro F1 | Notes |
|--------|-----------|-----------|--------|
| RandomForest | ~85% | 0.80 | Stable, interpretable baseline |
| XGBoost | ~87% | 0.82 | Best performance |
| LightGBM | ~86% | 0.81 | Lightweight, fast training |
| Logistic Regression | ~78% | 0.74 | Useful benchmark |

*(Exact scores vary depending on sampling and split seed)*

---

##  Author

**Sourav Kumar**  
📍 Bangalore, India  
📧 [souravmail003@gmail.com](mailto:souravmail003@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/sourav-kumar-5814341b8)  
💻 [GitHub](https://github.com/your-github-username)

---

##  Conclusion

This project demonstrates a full data science lifecycle — from **raw data understanding to deployment-ready ML pipeline** — applied to the real-world domain of workplace safety analytics.  
It highlights how combining **structured data** with **textual accident reports** can provide actionable insights to reduce workplace risk and improve industrial safety.

---

###  If you find this project useful, please give it a star on GitHub!
