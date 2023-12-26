# utils.py
"""
Useful functions for train and inference operations.
"""
import logging
import os
import subprocess

import pandas as pd


class OSUtils:
    """
    Useful utilities to perform operations: train and inference
    """

    def __init__(self):
        self.train_file = "train.csv"
        self.test_file = "test.csv"
        self.logger = logging.getLogger(self.__class__.__name__)

    def import_csv_to_dataframe(self, file_path):
        """
        Import a CSV file and convert it to a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            df (pd.DataFrame): The loaded data as a pandas DataFrame.
        """
        self.logger.info(f"Importing CSV file: {file_path}")
        df = pd.read_csv(file_path)
        return df

    def build_file_path(self, file_name):
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the file path
        file_path = os.path.join(current_dir, file_name)
        self.logger.info(f"Built file path: {file_path}")
        return file_path


class DVCUtils:
    """
    Basic operations with DVC repo
    """

    def __init__(self):
        self.remote_name = "hw2_dataset"
        self.remote_url = "gdrive://1I_ayLjvj_XboS1YcvcdFIIwiCMTQwYar?hl"
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_remote_dvc(self, remote_name, remote_url):
        """
        Add a new DVC remote.

        This method uses the 'dvc remote add' command
        to add a new DVC remote with the given name and URL.

        Args:
            remote_name (str): The name of the remote to add.
            remote_url (str): The URL of the remote to add.
        """
        self.logger.info(f"Adding DVC remote: {remote_name}")
        # Run the 'dvc remote add' command
        subprocess.run(
            ["dvc", "remote", "add", "-f", remote_name, remote_url], check=True
        )

    def pull_dvc(self):
        """
        Pull data from the DVC remote.

        This method uses the 'dvc pull' command to pull
        data from the DVC remote. It's useful when
         you want to download data files managed by DVC.
        """
        self.logger.info("Pulling data from DVC remote.")
        # Run the 'dvc pull' command
        subprocess.run(["dvc", "pull"], check=True)

    def commit_and_push_dvc(self):
        """
        Commit and push changes to the DVC remote.

        This method uses the 'dvc commit' command to
        save changes to DVC files,
        and the 'dvc push' command to push any
        changes in the data or pipeline to the DVC remote.
        """
        self.logger.info("Committing and pushing changes to DVC remote.")
        # Commit changes to DVC files
        subprocess.run(["dvc", "commit"], check=True)
        # Push changes to the DVC remote
        subprocess.run(["dvc", "push"], check=True)


#
# class OSUtils:
#     """
#     Useful utilities to perform operations: train and inference
#     """
#
#     def __init__(self):
#         self.train_file = "train.csv"
#         self.test_file = "test.csv"
#
#     @staticmethod
#     def import_csv_to_dataframe(file_path):
#         """
#         Import a CSV file and convert it to a pandas DataFrame.
#
#         Args:
#             file_path (str): The path to the CSV file.
#
#         Returns:
#             df (pd.DataFrame): The loaded data as a pandas DataFrame.
#         """
#         df = pd.read_csv(file_path)
#         return df
#
#     @staticmethod
#     def build_file_path(file_name):
#         # Get the current directory
#         current_dir = os.path.dirname(os.path.abspath(__file__))
#         # Construct the file path
#         file_path = os.path.join(current_dir, file_name)
#         return file_path
#
#
# class DVCUtils:
#     """
#     Basic operations with DVC repo
#     """
#
#     def __init__(self):
#         self.remote_name = "hw2_dataset"
#         self.remote_url = "gdrive://1I_ayLjvj_XboS1YcvcdFIIwiCMTQwYar?hl"
#
#     @staticmethod
#     def add_remote_dvc(remote_name, remote_url):
#         """
#         Add a new DVC remote.
#
#         This method uses the 'dvc remote add' command
#         to add a new DVC remote with the given name and URL.
#
#         Args:
#             remote_name (str): The name of the remote to add.
#             remote_url (str): The URL of the remote to add.
#         """
#         # Print the name of the remote
#         print(f"DVC Remote to use: {remote_name}")
#
#         # Run the 'dvc remote add' command
#         subprocess.run(
#             ["dvc", "remote", "add", "-f", remote_name, remote_url], check=True
#         )
#
#     @staticmethod
#     def pull_dvc():
#         """
#         Pull data from the DVC remote.
#
#         This method uses the 'dvc pull' command to pull
#         data from the DVC remote. It's useful when
#          you want to download data files managed by DVC.
#         """
#         # Run the 'dvc pull' command
#         subprocess.run(["dvc", "pull"], check=True)
#
#     @staticmethod
#     def commit_and_push_dvc():
#         """
#         Commit and push changes to the DVC remote.
#
#         This method uses the 'dvc commit' command to
#         save changes to DVC files,
#         and the 'dvc push' command to push any
#         changes in the data or pipeline to the DVC remote.
#         """
#         # Commit changes to DVC files
#         subprocess.run(["dvc", "commit"], check=True)
#
#         # Push changes to the DVC remote
#         subprocess.run(["dvc", "push"], check=True)
