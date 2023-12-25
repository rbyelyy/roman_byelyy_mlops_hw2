"""
Train Model
"""
import os
import pickle

import git
import hydra
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

import roman_byelyy_mlops_hw2.package.utils as ut


class TrainModel:
    """
    Main class for model training and data downloading
    """

    def __init__(self, cfg):
        """
        Initialize the TrainModel class with the given configuration.

        Args:
            cfg: A configuration object containing parameters for the model and preprocessing steps.
        """
        self.cfg = cfg
        self.data = None
        self.model = RandomForestRegressor(
            n_estimators=self.cfg.model.n_estimators,
            random_state=self.cfg.model.random_state,
            max_depth=self.cfg.model.max_depth,
            max_features=self.cfg.model.max_features,
        )
        self.os_utils = ut.OSUtils()

    def load_data(self):
        """
        Load the training data using DVC and OS utilities. The
        data is expected to be in CSV format.

        The method first checks if the file exists at the specified path.
        If the file exists, it is loaded into a DataFrame.
        If the file does not exist, a message is printed to the console.

        The method also sets up a DVC remote and pulls the data
        from the remote if necessary.
        """
        # Initialize DVC and OS utility instances
        dvc_instance = ut.DVCUtils()
        os_instance = ut.OSUtils()

        # Construct the file path
        file_path = ut.OSUtils.build_file_path(os_instance.train_file)

        # Set up a DVC remote and pull the data
        remote_name = dvc_instance.remote_name
        remote_url = dvc_instance.remote_url
        dvc_instance.add_remote_dvc(remote_name=remote_name, remote_url=remote_url)
        dvc_instance.pull_dvc()

        # Check if the file exists
        if os.path.isfile(file_path):
            print(f"Nice file for training is present: {file_path}")
            # If the file exists, load it into a DataFrame
            self.data = os_instance.import_csv_to_dataframe(file_path)
        else:
            print(f"Hm...I cannot find file: {file_path}")
            exit("if no training file then no prediction :-(")

    def preprocess_data(self) -> None:
        """
        Preprocess the data using the parameters defined
        in the Hydra configuration.
        This includes one-hot encoding categorical
        features, imputing missing values,
        and scaling numerical features.
        """
        # One-hot encode categorical features
        self.data = pd.get_dummies(self.data, columns=["ocean_proximity"])
        columns = self.data.columns

        # Create an instance of SimpleImputer with
        # the strategy defined in the Hydra config
        impute = SimpleImputer(strategy=self.cfg.imputer.strategy)

        # Apply SimpleImputer to the DataFrame
        self.data = pd.DataFrame(
            impute.fit_transform(self.data), columns=self.data.columns
        )

        # Create an instance of StandardScaler with
        # the parameters defined in the Hydra config
        with_mean = self.cfg.scaler.with_mean
        with_std = self.cfg.scaler.with_std
        scaler = StandardScaler(with_mean=with_mean, with_std=with_std)

        # Apply StandardScaler to the DataFrame
        df_scaled = scaler.fit_transform(self.data)

        # Convert the NumPy array back to a DataFrame
        self.data = pd.DataFrame(df_scaled, columns=columns)

    def train(self):
        try:
            # Get the git commit id
            repo = git.Repo(search_parent_directories=True)
            git_commit_id = repo.head.object.hexsha

            # Set up MLflow
            mlflow.set_tracking_uri("http://128.0.1.1:8080")
            mlflow.set_experiment("roman_byelyy_mlops_hw2")

            # Set up TensorBoard
            writer = SummaryWriter(log_dir="./runs")

            with mlflow.start_run(run_name="rbyelyy_run"):
                # Log hyperparameters and code version
                mlflow.log_param("test_size", self.cfg.split.test_size)
                mlflow.log_param("random_state", self.cfg.model.random_state)
                mlflow.log_param("git_commit_id", git_commit_id)

                # Check if data is loaded and preprocessed
                if self.data is None:
                    raise ValueError("Data is not loaded or preprocessed.")

                # Separate the features (X) and the target variable (y)
                X = self.data.drop("median_house_value", axis=1)
                y = self.data["median_house_value"]

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=self.cfg.split.test_size,
                    random_state=self.cfg.model.random_state,
                )

                train_losses = []
                val_losses = []

                for epoch in range(5):
                    # Train the model using the training dataset
                    self.model.fit(X_train, y_train)

                    # Compute MSE
                    y_pred_train = self.model.predict(X_train)
                    mse_train = mean_squared_error(y_train, y_pred_train)

                    # Add the MSE value to the list
                    train_losses.append(mse_train)

                    # Compute MSE on the validation dataset
                    y_pred_val = self.model.predict(X_test)
                    mse_val = mean_squared_error(y_test, y_pred_val)

                    # Add the MSE value to the list
                    val_losses.append(mse_val)

                    # Log MSE in TensorBoard
                    writer.add_scalar("Loss/train", mse_train, epoch)
                    writer.add_scalar("Loss/val", mse_val, epoch)

                # Create a plot of training and validation losses
                fig = plt.figure()
                plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
                plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val")
                plt.title("Training and Validation Losses")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()

                # Log the plot in TensorBoard
                writer.add_figure("Loss/plot", fig)

                # Compute MSE on the test dataset
                y_pred_test = self.model.predict(X_test)
                mse_test = mean_squared_error(y_test, y_pred_test)

                # Log MSE in MLflow
                mlflow.log_metric("mse", mse_test)

                # Compute the model's accuracy
                accuracy = self.model.score(X_test, y_test)

                # Print the model's accuracy
                print(f"Model accuracy: {accuracy}")

                # Log accuracy in MLflow
                mlflow.log_metric("accuracy", accuracy)

                # Log the model in MLflow
                mlflow.sklearn.log_model(self.model, "model")

        except Exception as e:
            print(f"An error occurred during the training process: {e}")

    def save_model(self):
        """
        Save the trained model to a file.

        The model is saved using pickle, which serializes the
        model object for future use. The filename is defined in
        the Hydra configuration.

        Raises:
            FileNotFoundError: If the directory specified in the
            Hydra config does not exist.
        """
        # Save the model to a file
        with open(self.cfg.output.model_file, "wb") as f:
            pickle.dump(self.model, f)

        # Save the model in MLflow
        mlflow.sklearn.log_model(self.model, "model")


@hydra.main(config_path="../configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    This function creates an instance of the TrainModel class and
    performs data loading, preprocessing, model training, and model saving.

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
