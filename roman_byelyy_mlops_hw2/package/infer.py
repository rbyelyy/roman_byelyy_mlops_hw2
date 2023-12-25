# infer.py
"""
Performing data downloading from DVC and applying model on a test data
"""
import os
import pickle

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import roman_byelyy_mlops_hw2.package.utils as ut


class InferModel:
    """
    Main class for making an inference for the model
    """

    def __init__(self):
        """
        Constructor for the InferModel class. Initializes all
        necessary attributes.
        """
        self.target_scaler = None  # Scaler for the target variable
        self.model = None  # The model we will use for predictions
        self.data = None  # The data we will make predictions on
        self.scaler = None  # Scaler for the features
        self.original_data = None  # Original data before preprocessing

    def load_data(self):
        """
        Loads the data for prediction. The data is loaded
        using DVC and OS utilities.
        """
        # Initialize DVC and OS utility instances
        dvc_instance = ut.DVCUtils()
        os_instance = ut.OSUtils()

        # Construct the file path
        # file_path = os_instance.test_file
        file_path = ut.OSUtils.build_file_path(os_instance.test_file)

        # Set up a DVC remote and pull the data
        remote_name = dvc_instance.remote_name
        remote_url = dvc_instance.remote_url
        dvc_instance.add_remote_dvc(remote_name=remote_name, remote_url=remote_url)
        dvc_instance.pull_dvc()

        # Check if the file exists
        if os.path.isfile(file_path):
            print(f"File is present: {file_path}")
            # If the file exists, load it into a DataFrame
            self.data = os_instance.import_csv_to_dataframe(file_path)
            self.original_data = self.data.copy()
            target = self.data["median_house_value"]
            self.target_scaler = StandardScaler()
            self.target_scaler.fit_transform(target.values.reshape(-1, 1))
            self.data = self.data.drop("median_house_value", axis=1)
        else:
            print(f"File is not present: {file_path}")

    def preprocess_data(self) -> None:
        """
        Preprocesses the data. This includes one-hot encoding
        categorical features, imputing missing values,
        and scaling numerical features.
        """
        # One-hot encode categorical features
        self.data = pd.get_dummies(self.data, columns=["ocean_proximity"])
        columns = self.data.columns

        # Create an instance of SimpleImputer with the
        # strategy defined in the Hydra config
        impute = SimpleImputer()

        # Apply SimpleImputer to the DataFrame
        self.data = pd.DataFrame(
            impute.fit_transform(self.data), columns=self.data.columns
        )

        # Create an instance of StandardScaler with
        # the parameters defined in the Hydra config
        self.scaler = StandardScaler()

        # Apply StandardScaler to the DataFrame
        df_scaled = self.scaler.fit_transform(self.data)

        # Convert the NumPy array back to a DataFrame
        self.data = pd.DataFrame(df_scaled, columns=columns)

    def load_model(self, model_name):
        """
        Loads the model from a .pkl file.

        Args:
            model_name (str): The name of the model
            file without the extension.
        """
        # Specify the path to your model file
        model_file = f"{model_name}.pkl"

        # Load the model from the file
        with open(model_file, "rb") as file:
            self.model = pickle.load(file)

    def predict(self):
        """
        Makes predictions based on the loaded model and data.

        Returns:
            numpy.ndarray: An array with the model's predictions.
        """
        predictions = self.model.predict(self.data)
        return predictions

    def inverse_scale_predictions(self, predictions):
        """
        Transforms the predictions back to the original scale.

        Args:
            predictions (numpy.ndarray): An array with
            the model's predictions.

        Returns:
            numpy.ndarray: An array with the predictions
            transformed back to the original scale.
        """
        predictions = predictions.reshape(-1, 1)
        return self.target_scaler.inverse_transform(predictions)

    @staticmethod
    def save_predictions(result, filename):
        """
        Saves the predictions to a .csv file.

        Args:
            result (pandas.DataFrame): A DataFrame
            with the predictions.
            filename (str): The name of the file
            where the predictions will be saved.
        """
        result.to_csv(filename, index=False)


def main() -> None:
    """
    Main entry point of the script.

    This function creates an instance of the TrainModel
    class and performs data loading, preprocessing, model training,
    and model saving.

    Raises:
        ValueError: If the data is not loaded or
        preprocessed before training.
    """
    # Create an instance of TrainModel with the Hydra configuration
    model = InferModel()

    # Load the data
    model.load_data()

    # Load the model
    model.load_model(model_name="model")

    # Preprocess the data
    model.preprocess_data()

    # Make predictions
    predictions = model.predict()

    # Inverse scale the predictions
    predictions = model.inverse_scale_predictions(predictions)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["Predictions"])

    # Concatenate the original data with the predictions
    result = pd.concat([model.original_data, predictions_df], axis=1)

    # Save the predictions to a csv file
    model.save_predictions(result, "predictions.csv")


if __name__ == "__main__":
    main()
