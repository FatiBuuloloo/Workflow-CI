import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Calories_Burnt_ScikitLearn")
mlflow.sklearn.autolog()

with mlflow.start_run():
    data_path = "calories_processed.csv" 
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {data_path}.")
    data = pd.read_csv(data_path)
    if "User_ID" in data.columns:
        data.drop("User_ID", axis=1, inplace=True)
    if "Gender" in data.columns and data["Gender"].dtype == 'object':
        data["Gender"] = data["Gender"].apply(lambda x: 1 if x == "male" else 0)
    data.dropna(inplace=True)
    X = data.drop("Calories", axis=1)
    y = data["Calories"]
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    
    features_to_transform = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
    features_to_transform = [col for col in features_to_transform if col in X.columns]
    if features_to_transform:
        pt = PowerTransformer(method='box-cox', standardize=True)
        train_X_trans = train_X.copy()
        test_X_trans = test_X.copy()
        train_X_trans[features_to_transform] = pt.fit_transform(train_X[features_to_transform])
        test_X_trans[features_to_transform] = pt.transform(test_X[features_to_transform])
    else:
        train_X_trans = train_X
        test_X_trans = test_X
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train_X_trans, train_y)
    y_pred = model.predict(test_X_trans)
    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    rmse = np.sqrt(mse)
    print(f"Hasil Evaluasi -> MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
    print("Training Selesai! Autolog telah merekam semua artefak ke MLflow.")