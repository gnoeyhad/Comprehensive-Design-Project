import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import numpy as np
import joblib

class MoisturePorePredictor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.df = self._load_and_process_data()
        self.best_model_moisture = None
        self.best_model_pore = None

    # 모델 학습을 위해 파일 불러오기 및 데이터셋 정리 
    def _load_data_from_json(self):
        data = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.data_dir, filename), 'r') as file:
                    file_data = json.load(file)
                    age = file_data['info']['age']
                    try:
                        moisture = file_data['equipment'].get('l_cheek_moisture', file_data['equipment']['l_cheek_moisture'])
                        pore = file_data['equipment'].get('l_cheek_pore', file_data['equipment']['l_cheek_pore'])
                    except:
                        moisture = file_data['equipment'].get('r_cheek_moisture', file_data['equipment']['r_cheek_moisture'])
                        pore = file_data['equipment'].get('r_cheek_pore', file_data['equipment']['r_cheek_pore'])
                    gender = file_data['info'].get('gender')
                    data.append({'Age': age, 'Moisture': moisture, 'Pore': pore, 'Gender_M': gender})
        return pd.DataFrame(data)

    # 결측치 제거 
    def _remove_outliers(self, df, column):
        Q1, Q3 = df[column].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        return df[~((df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR)))]

    def _load_and_process_data(self):
        df = self._load_data_from_json().dropna(subset=['Moisture', 'Pore', 'Gender_M'])
        df = self._remove_outliers(df, 'Moisture')
        df = self._remove_outliers(df, 'Pore')
        df['Pore'] = np.log1p(df['Pore'])
        return df
    

    # 모델 최적화 - XGB Regression 사용
    def _optimize_model(self, X_train, y_train):
        def xgb_evaluate(n_estimators, max_depth, learning_rate, subsample, colsample_bytree):
            model = XGBRegressor(
                n_estimators=int(n_estimators), max_depth=int(max_depth), learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=colsample_bytree, random_state=42
            )
            return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))

        optimizer = BayesianOptimization(
            f=xgb_evaluate, pbounds={
                'n_estimators': (50, 200), 'max_depth': (3, 10), 'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0), 'colsample_bytree': (0.6, 1.0)
            }, random_state=42
        )
        optimizer.maximize(init_points=10, n_iter=30)
        return optimizer.max['params']

    # 모델 학습 
    def train_models(self):
        X = pd.get_dummies(self.df[['Age', 'Gender_M']], drop_first=True)
        y_moisture, y_pore = self.df['Moisture'], self.df['Pore']
        X_train, X_test, y_train_moisture, y_test_moisture = train_test_split(X, y_moisture, test_size=0.2, random_state=42)
        _, _, y_train_pore, y_test_pore = train_test_split(X, y_pore, test_size=0.2, random_state=42)

        best_params_moisture = self._optimize_model(X_train, y_train_moisture)
        best_params_pore = self._optimize_model(X_train, y_train_pore)

        self.best_model_moisture = XGBRegressor(
            n_estimators=int(best_params_moisture['n_estimators']),
            max_depth=int(best_params_moisture['max_depth']),
            learning_rate=best_params_moisture['learning_rate'],
            subsample=best_params_moisture['subsample'],
            colsample_bytree=best_params_moisture['colsample_bytree'],
            random_state=42
        )
        self.best_model_pore = XGBRegressor(
            n_estimators=int(best_params_pore['n_estimators']),
            max_depth=int(best_params_pore['max_depth']),
            learning_rate=best_params_pore['learning_rate'],
            subsample=best_params_pore['subsample'],
            colsample_bytree=best_params_pore['colsample_bytree'],
            random_state=42
        )
        self.best_model_moisture.fit(X_train, y_train_moisture)
        self.best_model_pore.fit(X_train, y_train_pore)

    # 모델 학습 결과 저장
    def save_models(self, moisture_model_path='moisture_model.pkl', pore_model_path='data/pore_model.pkl'):
        joblib.dump(self.best_model_moisture, moisture_model_path)
        joblib.dump(self.best_model_pore, pore_model_path)
        print(f"Models saved to {moisture_model_path} and {pore_model_path}")

    # 모델 불러오기 
    def load_models(self, moisture_model_path='moisture_model.pkl', pore_model_path='data/pore_model.pkl'):
        self.best_model_moisture = joblib.load(moisture_model_path)
        self.best_model_pore = joblib.load(pore_model_path)
        print(f"Models loaded from {moisture_model_path} and {pore_model_path}")

    # 입력값으로 수분값, 모공값 예측 
    def predict(self, age, gender):
        if self.best_model_moisture is None or self.best_model_pore is None:
            raise ValueError("Models are not loaded or trained yet.")
        
        try:
            input_data = pd.DataFrame({'Age': [age], 'Gender_M': [gender]})
            input_data = pd.get_dummies(input_data, drop_first=True).reindex(columns=self.df[['Age', 'Gender_M']].drop_duplicates().columns, fill_value=0)

            moisture_pred = float(self.best_model_moisture.predict(input_data)[0])
            pore_pred = float(np.expm1(self.best_model_pore.predict(input_data)[0]))

            return {'Predicted Moisture': moisture_pred, 'Predicted Pore': pore_pred}

        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}

# 사용 예시
#predictor = MoisturePorePredictor('C:\\Users\\da010\\Downloads\\028.한국인 피부상태 측정 데이터\\3.개방데이터\\1.데이터\\Validation\\02.라벨링데이터\\VL\\All_json_file')
#predictor.train_models()
#predictor.save_models()  # 모델을 파일로 저장
# predictor.load_models()  # 저장된 모델 불러오기
# predictions = predictor.predict(age, gender)  # 예측 수행
