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
    ...
```

### Aggressiveness vs. PNL Plot
```python
def aggressiveness_vs_pnl():
    ...
```

### PNL Driver Analysis (Random Forest)
```python
def pnl_driver_analysis():
    ...
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
