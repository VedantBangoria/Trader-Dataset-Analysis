import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("output.csv")

def mostAggressiveTraders():
    traderAgressivenessScores = {}

    for row in df.itertuples(index=True):
        plcOverTransactions = (row.price_levels_consumed/row.transactions_per_day) if row.transactions_per_day != 0 else None
        lvlsPerTransaction = row.price_levels_per_transaction
        lvlsConsumedPerTransac_vw = row.price_levels_consumed_vw
        traderExitMeasure = row.mean_delta
        if(plcOverTransactions == None):
            traderAggressiveness = (lvlsPerTransaction * .15) + (lvlsConsumedPerTransac_vw * .17) + (traderExitMeasure * .15)
            traderAggressiveness += (row.transactions_per_day * .20) + (1/row.mean_time) * .13 + (math.log(row.markets_per_day + 1) * .20)
        else:
            traderAggressiveness = (plcOverTransactions * .18) + (lvlsPerTransaction * .15) + (lvlsConsumedPerTransac_vw * .12) + (traderExitMeasure * .10)
            traderAggressiveness += (row.transactions_per_day * .15) + (1/row.mean_time) * .10 + (math.log(row.markets_per_day + 1) * .20)

        traderAgressivenessScores[(row.trader, row.Index)] = traderAggressiveness


    sorted_traders = dict(sorted(traderAgressivenessScores.items(), key=lambda item: item[1], reverse=True))
    return sorted_traders

def aggressiveness_vs_pnl():
   
    # Map trader_id to pnl
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


print(df.describe())