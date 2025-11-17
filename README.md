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
