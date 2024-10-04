import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

#load the dataset
cal = fetch_california_housing()
df = pd.DataFrame(data=cal.data, columns=cal.feature_names)
df['price'] = cal.target
df.head()

#title of the app
st.title("California House Price Predictor for my Company")

#Data Overview
st.subheader("Data Overview")
st.dataframe(df.head(10))

#split the data into train and test
X = df.drop('price', axis=1)
Y = df['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#standardize the data
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

#model selection
st.subheader(" ### Select a model")

model = st.selectbox("Choose a model", ["linear Regression", "Ridge", "Lasso", "Elastic Net"])

#intialize the model
models = {
    "linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet()
}

# Select the model
selected_model = models[model]

#train the model
selected_model.fit(X_train_sc, Y_train)

#predicting the values
Y_pred = selected_model.predict(X_test_sc)

#evaluate the model
test_mse = mean_squared_error(Y_test, Y_pred)
st.write(f"Test MSE: {test_mse}")
test_mae = mean_absolute_error(Y_test, Y_pred)
st.write(f"Test MAE: {test_mae}")
test_r2 = r2_score(Y_test, Y_pred)
st.write(f"Test R2: {test_r2}")
test_rmse = np.sqrt(test_mse)
st.write(f"Test RMSE: {test_rmse}")

#Prompt the user to enter input values
st.write("Enter the input values to predict house price:")
user_input = {}

for feature in X.columns:
    user_input[feature] = st.number_input(feature)
    
df1 = pd.DataFrame([user_input])

#scale the user input
df1_sc = scaler.transform(df1)

#predict the house price
df1_pred = selected_model.predict(df1_sc)

#display predicted house price
st.write(f"Predicted house price: {df1_pred[0]*100000}" )

