# Sentiment Analysis on Product Reviews  

## 📌 Project Overview
This project focuses on **Sentiment Analysis** of product reviews to classify them as **positive** or **negative**.  
The dataset used is the [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews).  

### Goal
- Preprocess product reviews.  
- Convert text into numerical representation (TF-IDF & CountVectorizer).  
- Train classification models.  
- Compare performance of different models.  

---

## Dataset
- **Source**: Kaggle Amazon Reviews Dataset.  
- Contains millions of reviews labeled as positive (`__label__2`) or negative (`__label__1`).  
- For training and evaluation, we used a subset of 3M samples.  

---

##  Workflow
1. **Data Preprocessing**
   - Lowercasing
   - Stopword removal
   - Tokenization
2. **Feature Extraction**
   - TF-IDF Vectorizer
   - CountVectorizer
3. **Model Training**
   - Logistic Regression
   - Multinomial Naive Bayes
4. **Evaluation**
   - Accuracy
   - Precision, Recall, F1-Score
   - Confusion Matrix
   - Comparison Table & Visualization  

---

##  Results

| Model                | Accuracy |   F1   |
|----------------------|----------|--------|
| Logistic Regression + TF-IDF      | 0.8952   | 0.8989 |
| Naive Bayes + TF-IDF              | 0.8702   | 0.8749 |
| Logistic Regression + CountVector | 0.8955   | 0.8993 |
| Naive Bayes + CountVector         | 0.8679   | 0.8718 |

- **Best Model**: Logistic Regression with CountVectorizer  
  - Accuracy: **89.55%**
  - F1-score: **0.8993**

---

## Visualization
Bar chart comparison of models:

![Results Graph](results_comparison.png)

---

##  Tools & Libraries
- Python  
- Pandas, Numpy  
- NLTK / SpaCy  
- Scikit-learn  
- Matplotlib  

---

## Conclusion
- Logistic Regression performed best for sentiment analysis with both TF-IDF and CountVectorizer.  
- CountVectorizer slightly outperformed TF-IDF in this dataset.  
- Naive Bayes was simpler but less accurate compared to Logistic Regression.  

---

##  Future Work
- Try deep learning models (RNNs, LSTMs, or Transformers like BERT).  
- Perform hyperparameter tuning.  
- Explore more advanced preprocessing (lemmatization, bigrams).  
- Apply on a different dataset (IMDb, Twitter Sentiment).  

---

##  Author
- **Khloud Mohamed Ibrahem Ali**  
- Computer Science (AI Major) | Passionate about NLP, Machine Learning & Deep Learning  
- [LinkedIn](https://www.linkedin.com/in/kholoud-mohamed-07-I) | [GitHub](https://github.com/khloudmo)  

