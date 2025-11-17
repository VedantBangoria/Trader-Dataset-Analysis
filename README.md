# 📊 Trader Dataset Analysis

This repository contains a Python-based data analysis project exploring trader behavior, aggressiveness, profitability (PNL), and feature importance using machine learning and statistical methods.

---

## 🔎 Project Summary

This analysis performs three core actions:

### **1. Quantify Trader Aggressiveness**
A custom scoring function evaluates each trader using:
- Price levels consumed  
- Transactions per day  
- Mean time and delta-based risk indicators  
- Markets traded  
- Weighted metrics to reflect trading behavior  

The system outputs a ranked list of traders by aggressiveness.

---

### **2. Analyze Correlation Between Aggressiveness and PNL**
- Builds a scatterplot with **aggressiveness score vs. trader PNL**  
- Fits a **Least Squares Regression Line (LSRL)**  
- Determines correlation direction and slope  

**Regression Line:**

```
y = 1.63e-02(x) + 11.43
```

---

### **3. Rank Features Driving Trader PNL**
A **Random Forest Regressor** ranks the importance of trading activity features in predicting PNL.

**Top Features Identified:**
1. trader_volume  
2. price_levels_consumed_vw  
3. mean_delta  
4. markets_per_day  
5. price_levels_consumed  
6. std_delta  
7. price_levels_per_transaction  
8. transactions_per_day  

Model performance:
- **R² ≈ 0.127**  
- **RMSE ≈ 6870.28**

---

## 📁 Dataset

The dataset (`output.csv`) contains:
- **604,578 rows**
- **41 columns**
- Trader metrics (volume, time, PNL, price levels)
- Tag/topic distribution features
- No missing values after replacing `inf` and `-inf` and applying `.fillna(0)`

Place the CSV file in the project root before running the notebook.

---

## 🧠 Code Structure

### **Imports**
```python
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
```

### Aggressiveness Calculation
```python
def mostAggressiveTraders():
    traderAgressivenessScores = {}

    for row in df.itertuples(index=True):
        #higher fraction means more price levels consumed per transaction, indicating larger orders and therefore more aggressiveness
        plcOverTransactions = (row.price_levels_consumed/row.transactions_per_day) if row.transactions_per_day != 0 else None
        lvlsPerTransaction = row.price_levels_per_transaction
        lvlsConsumedPerTransac_vw = row.price_levels_consumed_vw

        #distance from midpoint price indicates how much a trader is willing to sacrifice, measuring their riskiness and therefore aggression
        traderExitMeasure = row.mean_delta
        
        #account for zero division errors
        if(plcOverTransactions == None):
            traderAggressiveness = (lvlsPerTransaction * .15) + (lvlsConsumedPerTransac_vw * .17) + (traderExitMeasure * .15)
            traderAggressiveness += (row.transactions_per_day * .20) + (1/row.mean_time) * .13 + (math.log(row.markets_per_day + 1) * .20)
        else:
            traderAggressiveness = (plcOverTransactions * .18) + (lvlsPerTransaction * .15) + (lvlsConsumedPerTransac_vw * .12) + (traderExitMeasure * .10)
            traderAggressiveness += (row.transactions_per_day * .15) + (1/row.mean_time) * .10 + (math.log(row.markets_per_day + 1) * .20)

        traderAgressivenessScores[(row.trader, row.Index)] = traderAggressiveness


    sorted_traders = dict(sorted(traderAgressivenessScores.items(), key=lambda item: item[1], reverse=True))
    return sorted_traders

```

### Aggressiveness vs. PNL Plot
```python
def aggressiveness_vs_pnl():
    trader_to_pnl = {row.trader: row.trader_pnl for row in df.itertuples(index=True)}
    
    x = []
    y = []

    for trader_tuple, score in mostAggressiveTraders().items():
        trader_id = trader_tuple[0]  # extract trader_id from the key tuple
        pnl = trader_to_pnl.get(trader_id, 0)  # default to 0 if trader not found
        x.append(pnl)
        y.append(score)
    
    x = np.array(x)
    y = np.array(y)

    # Compute linear regression coefficients
    m, b = np.polyfit(x, y, 1)  # y = m*x + b

    # Plot scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.3, color='blue', label='Traders')

    # Plot regression line
    plt.plot(x, m*x + b, color='red', linewidth=2, label=f'Regression line: y={m:.2e}x + {b:.2f}')

    # Set bounds
    plt.xlim(-1.5e6, 1.5e6)
    plt.ylim(0, 100000)

    # Labels
    plt.xlabel("Trader PnL")
    plt.ylabel("Aggressiveness Score")
    plt.title("Trader Aggressiveness vs PnL with Regression Line")
    plt.legend()
    plt.show()
```

### PNL Driver Analysis (Random Forest)
```python
def pnl_driver_analysis():
    features = [
        "trader_volume",
        "transactions_per_day",
        "markets_per_day",
        "price_levels_consumed",
        "price_levels_per_transaction",
        "price_levels_consumed_vw",
        "mean_delta",
        "std_delta"
    ]
    X = df[features].fillna(0)
    y = df["trader_pnl"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Random Forest regressor
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importances
    importances = rf.feature_importances_
    feature_importance_dict = {feat: imp for feat, imp in zip(features, importances)}

    # Sort features by importance
    sorted_features = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    print("Random Forest Feature Importances:")
    for feat, imp in sorted_features.items():
        print(f"{feat}: {imp:.4f}")

    print(f"\nR^2: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
```

---

## 📈 Sample Outputs

### **Top 5 Most Aggressive Traders**
```
('0x9d84cE0306F8551e02EFef1680475Fc0f1dC1344', 245942) → 92503.48
('0x96B59F71f635da5dA031e3E93448C54Fe226f5E7', 498158) → 65934.49
('0x715e4430442e3c7bf152041ac5A1A9D9B234E9df', 222297) → 49851.33
('0x3Cf3E8d5427aED066a7A5926980600f6C3Cf87B3', 549304) → 49404.40
('0x7C3Db723F1D4d8cB9C550095203b686cB11E5C6B', 211461) → 44406.51
```

### **Feature Importances**
```
trader_volume:                0.4123
price_levels_consumed_vw:     0.1165
mean_delta:                   0.1020
markets_per_day:              0.0855
price_levels_consumed:        0.0764
std_delta:                    0.0732
price_levels_per_transaction: 0.0682
transactions_per_day:         0.0658
```

---

## 🚀 How to Run the Project

Clone the repo:
```bash
git clone https://github.com/<your-username>/<your-repo>.git
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Add **output.csv** to the main directory.

Run the notebook:
```bash
jupyter notebook
```

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook

---

## 📌 Future Enhancements
- Add K-means clustering for trader segmentation  
- Train additional ML models to boost prediction accuracy  
- Build a live dashboard using Flask or FastAPI  
- Add model versioning + experiment tracking  

---

## 📄 License
Distributed under the MIT License.

---
