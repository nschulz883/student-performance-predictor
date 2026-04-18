

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score



# DATA CLEANING 

#open file 
filepath = input("enter a file ")
df = pd.read_csv(filepath)

#gets shape 
original_shape = df.shape


#gets columns and numbers 
numeric_columns = df.select_dtypes(include = np.number).columns
categorical_cols = df.select_dtypes(exclude = np.number).columns


#full data cleaning missing value handling fills with means and modes 

for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

df = df.drop_duplicates()

#make same copy of file for each 

df.to_csv("cleaned_data.csv")
print("data cleaning successfully completed... '\n")

print("running moduels now... '\n")
# Target setting

target = "Final_Grade"

X = df.drop(target, axis = 1)
Y = df[target]

#train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)



#Models

#LINEAR REGRESSION, 
lin_reg = LinearRegression()
# DECISION TREE REGRESSION 
dec_tree = DecisionTreeRegressor()
# RANDOM FOREST REGRESSOR
random = RandomForestRegressor()

#data fitting 

lin_reg.fit(X_train, y_train)
dec_tree.fit(X_train, y_train)
random.fit(X_train, y_train)

lin_reg_pred = lin_reg.predict(X_test)
dec_tree_pred = dec_tree.predict(X_test)
random_pred = random.predict(X_test)


# RMSE R**2 calculation 

def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> RMSE: {rmse:.2f}, R2: {r2:.2f}")
    return rmse, r2

lin_reg_rmse, lin_reg_r2 = evaluate("Linear Regression", y_test, lin_reg_pred)
dec_tree_rmse, dec_tree_r2 = evaluate("Decision Tree", y_test, dec_tree_pred)
random_rmse, random_r2 = evaluate("Random Forest", y_test, random_pred)



#PLOTS 

# Actual vs Predicted
plt.scatter(y_test, lin_reg_pred, label="Linear (red)")
plt.scatter(y_test, dec_tree_pred, label="Tree (blue)")
plt.scatter(y_test, random_pred, label="Forest (green)")
plt.plot(y_test, y_test, color="black", label="Actual")
plt.legend()
plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")
plt.show()

# R2 bar chart 
models = ["Linear", "Tree", "Forest"]
scores = [lin_reg_r2, dec_tree_r2, random_r2]

plt.bar(models, scores)
plt.title("Model R2 Comparison")
plt.ylabel("R2 Score")
plt.show()

# Feature importance (Random Forest)
importances = random.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=45)
plt.show()

