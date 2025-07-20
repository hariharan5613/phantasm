import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression,  Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


df = pd.read_csv("milk_production.csv")
print(df.head())


print(df.isnull().sum())


X = df.drop("Milk_Production", axis=1)
y = df["Milk_Production"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "LinearRegression": LinearRegression(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR()
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append((name, mse, r2))
    print(f"{name} -> MSE: {mse:.2f}, R2: {r2:.2f}")


results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2"])
results_df = results_df.sort_values(by="R2", ascending=False)
print("\nModel Comparison:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x="R2", y="Model", data=results_df, palette="viridis")
plt.title("Regression Model Performance on Milk Production")
plt.xlabel("R2 Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()
