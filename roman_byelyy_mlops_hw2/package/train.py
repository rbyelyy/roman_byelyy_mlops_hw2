import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_california_housing
import pickle

class TrainModel:
    def __init__(self):
        california_housing = fetch_california_housing(as_frame=True)
        self.data = california_housing.frame
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        # Здесь вы можете добавить код для предварительной обработки данных
        pass

    def train(self):
        X = self.data.drop('MedHouseVal', axis=1)
        y = self.data['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print(f"Model accuracy: {self.model.score(X_test, y_test)}")

    def save_model(self):
        with open('model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

if __name__ == '__main__':
    model = TrainModel()
    model.preprocess_data()
    model.train()
    model.save_model()
