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
y = 1.63e-02(x) + 11.43


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

`output.csv` contains:
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
