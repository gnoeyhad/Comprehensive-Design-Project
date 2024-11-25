import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class SkinPredictor:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.df = self.load_data_from_json()
        self.best_model_moisture = None
        self.best_model_pore = None

    # JSON 데이터를 로드하는 메서드
    def load_data_from_json(self):
        data = []
        for filename in os.listdir(self.data_directory):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_directory, filename), 'r') as file:
                    file_data = json.load(file)

                    age = file_data['info']['age']
                    val_l = file_data['equipment'].get('l_cheek_moisture')
                    
                    if val_l is not None:
                        moisture = file_data['equipment']['l_cheek_moisture']
                        pore = file_data['equipment']['l_cheek_pore']
                    else:
                        moisture = file_data['equipment']['r_cheek_moisture']
                        pore = file_data['equipment']['r_cheek_pore']

                    gender = file_data['info'].get('gender', None)
                    sensitive = file_data['info'].get('sensitive', None)

                    data.append({
                        'Age': age,
                        'Moisture': moisture,
                        'Pore': pore,
                        'Gender': gender,
                        'Sensitive': sensitive
                    })
        df = pd.DataFrame(data)
        df = df.dropna(subset=['Moisture', 'Pore', 'Gender', 'Sensitive'])
        return df

    # 데이터 준비 및 모델 학습 메서드
    def prepare_and_train_models(self):
        df = self.df
        X = df[['Age', 'Gender', 'Sensitive']]
        X = pd.get_dummies(X, drop_first=True)
        y_moisture = df['Moisture']
        y_pore = df['Pore']

        # 데이터 분할
        X_train, X_test, y_train_moisture, y_test_moisture = train_test_split(X, y_moisture, test_size=0.2, random_state=42)
        _, _, y_train_pore, y_test_pore = train_test_split(X, y_pore, test_size=0.2, random_state=42)

        # XGBoost 모델 설정
        xgb_model = XGBRegressor()
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }

        # Moisture 모델 학습
        grid_search_moisture = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
        grid_search_moisture.fit(X_train, y_train_moisture)
        self.best_model_moisture = grid_search_moisture.best_estimator_

        # Pore 모델 학습
        grid_search_pore = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1)
        grid_search_pore.fit(X_train, y_train_pore)
        self.best_model_pore = grid_search_pore.best_estimator_

    # 예측 메서드
    def predict_values(self, age, gender, sensitive):
        input_data = pd.DataFrame({'Age': [age], 'Gender': [gender], 'Sensitive': [sensitive]})
        input_data = pd.get_dummies(input_data, columns=['Gender'], drop_first=True)
        
        if 'Gender_M' not in input_data.columns:
            input_data['Gender_M'] = 0

        input_data = input_data[['Age', 'Sensitive', 'Gender_M']]

        moisture_pred = self.best_model_moisture.predict(input_data)
        pore_pred = self.best_model_pore.predict(input_data)

        return moisture_pred[0], pore_pred[0]

    # 성능 평가 및 시각화 메서드
    def evaluate_and_plot(self, X_test, y_test_moisture, y_test_pore):
        # Moisture 예측 결과
        y_pred_moisture = self.best_model_moisture.predict(X_test)
        mse_moisture = mean_squared_error(y_test_moisture, y_pred_moisture)
        r2_moisture = r2_score(y_test_moisture, y_pred_moisture)

        # Pore 예측 결과
        y_pred_pore = self.best_model_pore.predict(X_test)
        mse_pore = mean_squared_error(y_test_pore, y_pred_pore)
        r2_pore = r2_score(y_test_pore, y_pred_pore)

        print(f"Moisture - MSE: {mse_moisture}, R2: {r2_moisture}")
        print(f"Pore - MSE: {mse_pore}, R2: {r2_pore}")

        # 시각화
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].scatter(X_test['Age'], y_test_moisture, label='Actual Moisture', alpha=0.6)
        axs[0].scatter(X_test['Age'], y_pred_moisture, label='Predicted Moisture', alpha=0.6)
        axs[0].legend()

        axs[1].scatter(X_test['Age'], y_test_pore, label='Actual Pore', alpha=0.6)
        axs[1].scatter(X_test['Age'], y_pred_pore, label='Predicted Pore', alpha=0.6)
        axs[1].legend()

        plt.show()

# 예시 코드 (다른 파일에서 사용)
# from skin_predictor import SkinPredictor
# predictor = SkinPredictor('C:\\path\\to\\your\\data')
# predictor.prepare_and_train_models()
# moisture, pore = predictor.predict_values(40, 'F', 1)
# print(f"Predicted Moisture: {moisture}, Predicted Pore: {pore}")
