import os
import pickle
import utils
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from omegaconf import DictConfig, OmegaConf
import hydra

class TrainModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = None
        self.model = RandomForestRegressor(n_estimators=self.cfg.model.n_estimators,
                                           random_state=self.cfg.model.random_state,
                                           max_depth=self.cfg.model.max_depth,
                                           max_features=self.cfg.model.max_features)
        self.os_utils = utils.OSUtils()

    @staticmethod
    def import_csv_to_dataframe(file_path):
        """
        Import a CSV file and convert it to a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            df (pd.DataFrame): The loaded data as a pandas DataFrame.
        """
        df = pd.read_csv(file_path)
        return df

    def load_data(self):
        """
        Load the training data using DVC and OS utilities. The data is expected to be in CSV format.

        The method first checks if the file exists at the specified path. If the file exists, it is loaded into a
        DataFrame.
        If the file does not exist, a message is printed to the console.

        The method also sets up a DVC remote and pulls the data from the remote if necessary.
        """
        # Initialize DVC and OS utility instances
        dvc_instance = utils.DVCUtils()
        os_instance = utils.OSUtils()

        # Construct the file path
        file_path = os_instance.train_file

        # Set up a DVC remote and pull the data
        dvc_instance.add_remote_dvc(remote_name=dvc_instance.remote_name, remote_url=dvc_instance.remote_url)
        dvc_instance.pull_dvc()

        # Check if the file exists
        if os.path.isfile(file_path):
            print(f'File is present: {file_path}')
            # If the file exists, load it into a DataFrame
            self.data = self.import_csv_to_dataframe(file_path)
        else:
            print(f'File is not present: {file_path}')

    def preprocess_data(self) -> None:
        """
        Preprocess the data using the parameters defined in the Hydra configuration.
        This includes one-hot encoding categorical features, imputing missing values,
        and scaling numerical features.
        """
        # One-hot encode categorical features
        self.data = pd.get_dummies(self.data, columns=['ocean_proximity'])
        columns = self.data.columns

        # Create an instance of SimpleImputer with the strategy defined in the Hydra config
        impute = SimpleImputer(strategy=self.cfg.imputer.strategy)

        # Apply SimpleImputer to the DataFrame
        self.data = pd.DataFrame(impute.fit_transform(self.data), columns=self.data.columns)

        # Create an instance of StandardScaler with the parameters defined in the Hydra config
        scaler = StandardScaler(with_mean=self.cfg.scaler.with_mean, with_std=self.cfg.scaler.with_std)

        # Apply StandardScaler to the DataFrame
        df_scaled = scaler.fit_transform(self.data)

        # Convert the NumPy array back to a DataFrame
        self.data = pd.DataFrame(df_scaled, columns=columns)

    def train(self):
        """
        Train the model using the preprocessed data.

        The data is split into training and testing sets. The split ratio and random state are defined in the Hydra
        configuration.
        The model is then trained using the training set and the accuracy of the model is printed.

        Raises:
            ValueError: If the data is not loaded or preprocessed before training.
        """
        # Check if data is loaded and preprocessed
        if self.data is None:
            raise ValueError("Data is not loaded or preprocessed.")

        # Separate the features (X) and the target variable (y)
        X = self.data.drop("median_house_value", axis=1)
        y = self.data["median_house_value"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.cfg.split.test_size, random_state=self.cfg.model.random_state
        )

        # Train the model using the training set
        self.model.fit(X_train, y_train)

        # Print the accuracy of the model on the testing set
        print(f"Model accuracy: {self.model.score(X_test, y_test)}")

    def save_model(self):
        """
        Save the trained model to a file.

        The model is saved using pickle, which serializes the model object for future use. The filename is defined in
        the Hydra configuration.

        Raises:
            FileNotFoundError: If the directory specified in the Hydra config does not exist.
        """
        # Save the model to a file
        with open(self.cfg.output.model_file, "wb") as f:
            pickle.dump(self.model, f)


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    This function creates an instance of the TrainModel class and performs data loading, preprocessing, model training,
    and model saving.

    Args:
        cfg (DictConfig): The Hydra configuration object.

    Raises:
        ValueError: If the data is not loaded or preprocessed before training.
    """
    # Create an instance of TrainModel with the Hydra configuration
    model = TrainModel(cfg)

    # Load the data
    model.load_data()

    # Preprocess the data
    model.preprocess_data()

    # Train the model
    model.train()

    # Save the trained model
    model.save_model()


if __name__ == "__main__":
    main()
