# 🎰 Casino ML Risk Analysis

A modular machine learning system analyzing **Blackjack, Poker, and Roulette** to study **predictability, strategy impact, and risk** in stochastic environments.

---

## 📌 Overview

This project explores how machine learning performs across different types of uncertainty:

* **Blackjack** → strategy-driven
* **Poker** → partially observable, multi-agent
* **Roulette** → purely random

The goal is to evaluate:

* when ML works
* when it fails
* and how strategy affects long-term outcomes

---

## 🧠 Key Results

* ♠️ **Poker Accuracy:** ~98.4% → strong structural patterns
* 🃏 **Blackjack Accuracy:** ~75.9% → strategy-dependent
* 🎲 **Roulette Accuracy:** ~2.5% ≈ random baseline → no predictability

### 💰 Simulation Insights

* Blackjack (basic strategy): **EV ≈ -0.02**
* Blackjack (random strategy): **EV ≈ -0.44**

👉 Strategy significantly reduces loss

---

## ⚙️ System Architecture

The project is fully modular:

```id="l6xj8g"
.
├── data_loading.py        # Efficient loading (chunked for large datasets)
├── preprocessing.py       # Cleaning and transformations
├── feature_engineering.py # Domain-specific feature creation
├── models.py              # ML model training (LR, RF, etc.)
├── evaluation.py          # Metrics and comparisons
├── simulation.py          # Monte Carlo simulations
├── utils.py               # Helper functions
├── main.py                # End-to-end pipeline
└── README.md
```

---

## 🔍 Methodology

### 1. Data Processing

* Large-scale Blackjack dataset (millions of hands)
* Roulette sequence data (~100k spins)
* Real poker hand-history logs (parsed from raw text)

---

### 2. Feature Engineering

* **Blackjack:** hand strength, dealer card, action encoding
* **Poker:** betting patterns, aggression score, positional features
* **Roulette:** streaks, rolling distributions, entropy

---

### 3. Modeling

Trained and compared:

* Logistic Regression
* Random Forest

Each game uses a **separate pipeline** tailored to its structure.

---

### 4. Simulation

Monte Carlo simulations used to evaluate:

* expected value (EV)
* variance
* strategy performance
* risk of loss

---

## 📊 Results Summary

| Game      | Model Accuracy | Insight                      |
| --------- | -------------- | ---------------------------- |
| Poker     | ~98.4%         | Strong patterns, predictable |
| Blackjack | ~75.9%         | Strategy matters             |
| Roulette  | ~2.5%          | Pure randomness              |

---

## 💡 Key Insights

* Machine learning performs well in **structured decision environments**
* Performance degrades in **high randomness systems**
* Strategy significantly impacts expected value in Blackjack
* Some systems (like Roulette) are fundamentally **unpredictable**

---

## 🚀 How to Run

### 1. Clone repository

```bash id="ap3u9p"
git clone https://github.com/MadhavKathuria826/casino-ml-risk-analysis.git
cd casino-ml-risk-analysis
```

### 2. Install dependencies

```bash id="jzzhlc"
pip install numpy pandas scikit-learn matplotlib
```

### 3. Run pipeline

```bash id="sc6prq"
python main.py
```

---

## 📦 Datasets

Due to size constraints, datasets are not included.

* Blackjack dataset (large-scale)
* Poker hand histories (text logs)
* Roulette dataset (~100k spins)

👉 Place datasets in appropriate directories before running.

---

## 📈 Outputs

Generated outputs include:

* `metrics_summary.csv`
* `model_comparisons.csv`
* `simulation_results.json`
* performance plots (accuracy, ROI, distributions)

---

## 🎯 Future Improvements

* Reinforcement learning for strategy optimization
* Sequence models (LSTM/Transformers) for poker
* Advanced probabilistic modeling
* Real-time simulation dashboard

---

## 👨‍💻 Author

**Madhav Kathuria**
B.Tech CSE, South Asian University

---

## ⭐ If you found this useful

Star the repo ⭐ — it helps!
