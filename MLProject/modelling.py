import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()
data_path = "calories_processed.csv" 
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File dataset tidak ditemukan di: {data_path}.")
data = pd.read_csv(data_path)
if "User_ID" in data.columns:
    data.drop("User_ID", axis=1, inplace=True)
X = data.drop("Calories", axis=1)
y = data["Calories"]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_X, train_y)
mlflow.sklearn.log_model(model, "model")
y_pred = model.predict(test_X)

mse = mean_squared_error(test_y, y_pred)
mae = mean_absolute_error(test_y, y_pred)
rmse = np.sqrt(mse)

print(f"Hasil Evaluasi -> MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
print("Training Selesai! MLflow Projects berhasil merekam model steril Anda.")