import os
import pickle
import utils
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class TrainModel:
    def __init__(self):
        self.data = None
        self.model = RandomForestRegressor()
        self.os_utils = utils.OSUtils()


    def import_csv_to_dataframe(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def load_data(self):
        dvc_instance = utils.DVCUtils()
        os_instance = utils.OSUtils()
        file_path = 'roman_byelyy_mlops_hw2/package/' + os_instance.train_file
        # dvc_instance.add_remote_dvc(remote_name=dvc_instance.remote_name, remote_url=dvc_instance.remote_url)
        # dvc_instance.pull_dvc()
        if os.path.isfile(file_path):
            print(f'File is present: {file_path}')
        else:
            print(f'File is not present: {file_path}')
        self.data =  self.import_csv_to_dataframe(file_path)

    def preprocess_data(self):
        self.data = pd.get_dummies(self.data, columns=['ocean_proximity'])
        columns = self.data.columns

        # Создать экземпляр SimpleImputer
        imputer = SimpleImputer(strategy='mean')  # или 'median', 'most_frequent'

        # Применить SimpleImputer к DataFrame
        self.data = pd.DataFrame(imputer.fit_transform(self.data), columns=self.data.columns)
        
        # Создать экземпляр StandardScaler
        scaler = StandardScaler()

        # Применить StandardScaler к DataFrame
        df_scaled = scaler.fit_transform(self.data)

        # Преобразовать массив NumPy обратно в DataFrame
        self.data = pd.DataFrame(df_scaled, columns=columns)

    
    def train(self):
        X = self.data.drop("median_house_value", axis=1)
        y = self.data["median_house_value"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        print(f"Model accuracy: {self.model.score(X_test, y_test)}")

    def save_model(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)


if __name__ == "__main__":
    model = TrainModel()
    model.load_data()
    model.preprocess_data()
    model.train()
    # model.save_model()
    # Загрузить данные из CSV-файла

