import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pickle

class LightGBMmodel:
    def __init__(self, params=None, model_path="app/models/lightgbm_model_100r.pkl"):
        self.params = params if params else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 50, 
            'max_depth': 10,
            'seed': 42
        }
        self.model_path = model_path
        self.model = None
    
    def load_data(self, file_path, features, target, test_size=0.2, random_state=42):
        df = pd.read_csv(file_path)
        X = df[features]
        y = df[target]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val
    
    def train(self, X_train, y_train, X_val, y_val):
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=300
        )
        print('Train model Accomplished!')

    def evaluate(self, X_val, y_val):
        if not self.model:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Evaluation Results:\nMSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return mse, mae, r2
    
    def save_model(self):
        # Save the trained model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.model_path}!")

    def load_model(self):
        # Load a trained model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {self.model_path}!")

    def predict(self, X):
        if not self.model:
            raise ValueError("Model not loaded! Please load or train the model before predicting.")
        
        features_df = pd.DataFrame([features])
        return self.model.predict(features_df)

if __name__ == "__main__":
    lgbm_model = LightGBMmodel()
    
    # define path and target
    fp = 'app/utils/results/features_with_aiscores.csv'
    features = ['cosine_similarity', 'jaccard_similarity', 'levenshtein_similarity']
    target = 'plagiarism_score'

    print("Load dan Target sudah di set!")

    X_train, X_val, y_train, y_val = lgbm_model.load_data(fp, features, target)

    lgbm_model.train(X_train, y_train, X_val, y_val)

    lgbm_model.evaluate(X_val, y_val)

    # Save the model
    lgbm_model.save_model()






    

