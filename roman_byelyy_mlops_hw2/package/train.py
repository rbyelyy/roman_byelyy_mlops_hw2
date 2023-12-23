# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

class TrainModel:
    def __init__(self):
        iris = load_iris()
        self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.data['target'] = iris.target
        self.model = RandomForestClassifier()

    def preprocess_data(self):
        # Здесь вы можете добавить код для предварительной обработки данных
        pass

    def train(self):
        X = self.data.drop('target', axis=1)
        y = self.data['target']
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