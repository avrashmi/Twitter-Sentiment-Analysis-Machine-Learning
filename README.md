
# Twitter Sentiment Analysis | Live Tweets & NLP

### Project Overview
This project performs **Sentiment Analysis on real-time tweets** fetched directly from Twitter using the **Tweepy API**.  
The goal is to analyze public opinion by classifying tweets as **Positive**, **Negative**, or **Neutral** using **Natural Language Processing (NLP)** techniques.

It demonstrates a full workflow — from **data collection** to **sentiment classification** and **visualization**, showcasing how machine learning and NLP can be applied to real social media data.

---

## Features
- Fetch **live tweets** using Twitter API (via Tweepy)
- **Clean and preprocess** text (remove emojis, URLs, hashtags, etc.)
- Apply **VADER** and **TextBlob** for sentiment scoring
- Visualize sentiment distribution using **Matplotlib** & **Seaborn**
- Train and evaluate ML classifiers (Naive Bayes, Logistic Regression, etc.)

---

## Tech Stack
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python 3.x |
| API | Tweepy (Twitter API v2/v1.1) |
| NLP & Sentiment | NLTK (VADER), TextBlob |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas, NumPy |

---

## Installation & Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Get Twitter API credentials

* Create a developer account at [developer.twitter.com](https://developer.twitter.com/)
* Create a project/app and generate:

  * `API_KEY`
  * `API_SECRET_KEY`
  * `ACCESS_TOKEN`
  * `ACCESS_TOKEN_SECRET`
* Add them to a `.env` file or directly in your script.

### Run the project

```bash
python sentiment_analysis.py
```

---

## Data Preprocessing Steps

* Remove URLs, mentions (@username), hashtags (#tag), and emojis.
* Convert text to lowercase.
* Tokenize and remove stopwords.
* Prepare cleaned text for sentiment analysis.

---

## Sentiment Analysis Techniques

### **VADER (Valence Aware Dictionary and sentiment Reasoner)**

* Rule-based sentiment analyzer tuned for social media text.
* Returns **compound score** ∈ [-1, 1]:

  * > 0 → Positive
  * < 0 → Negative
  * = 0 → Neutral

### **TextBlob**

* Provides **polarity** (−1 to +1) and **subjectivity** (0 to 1).
* Used to cross-verify or combine with VADER results.

---

## Visualizations

* **Pie Chart** of sentiment distribution.
* **WordCloud** for positive and negative words.
  

Example Output:

```
Positive: 😊😊😊😊😊
Negative: 😠😠
Neutral: 😐😐😐
```

---

## Machine Learning Classifier

We can extend this project to include ML models like:

* Logistic Regression
* Naive Bayes
* Support Vector Machines (SVM)
* Random Forest

Train the model using TF-IDF or Bag-of-Words features extracted from labeled tweets.

---

## Real-World Use Cases

* Brand reputation monitoring
* Political sentiment tracking
* Event popularity and audience mood
* Product feedback analysis

---

## Project Structure

```
twitter-sentiment-analysis/
│
├── sentiment_analysis.py       # Main script
├── twitter_api.py              # Tweepy API connection
├── preprocessing.py             # Cleaning functions

├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation
└── data/
    ├── tweets.csv               # Collected tweets
    └── cleaned_tweets.csv       # Preprocessed data
```

---

## Sample Results

| Sentiment | Count | Percentage |
| --------- | ----- | ---------- |
| Positive  | 523   | 52.3%      |
| Neutral   | 302   | 30.2%      |
| Negative  | 175   | 17.5%      |

---

## Future Improvements

* Integrate a **real-time dashboard** using Streamlit or Dash.
* Add **Deep Learning** model (LSTM / BERT) for advanced classification.
* Perform **topic modeling** (LDA) to find trending themes.
* Add **language detection & translation** for non-English tweets.

---
```
