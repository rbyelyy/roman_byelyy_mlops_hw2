import pickle
import os
import dvc.api
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import subprocess

class OSUtils:
    def __init__(self):
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
    
    @staticmethod
    def check_file_exists(filename):
        # Получить полный путь к текущему файлу
        current_file_path = os.path.realpath(__file__)
        current_directory = os.path.dirname(current_file_path)
        print(current_file_path)
        # return os.path.isfile(current_directory + '/' + filename)

class DVCUtils:
    def __init__(self):
        self.remote_name = 'hw2_dataset'
        self.remote_url = 'gdrive://1I_ayLjvj_XboS1YcvcdFIIwiCMTQwYar?hl'

    @staticmethod
    def add_remote_dvc(remote_name, remote_url):
        """
        Add a new DVC remote.

        This method uses the 'dvc remote add' command to add a new DVC remote with the given name and URL.

        Args:
            remote_name (str): The name of the remote to add.
            remote_url (str): The URL of the remote to add.
        """
        # Print the name of the remote
        print(remote_name)

        # Run the 'dvc remote add' command
        subprocess.run(["dvc", "remote", "add", "-f", remote_name, remote_url], check=True)

    @staticmethod
    def pull_dvc():
        """
        Pull data from the DVC remote.

        This method uses the 'dvc pull' command to pull data from the DVC remote. It's useful when you want to download data files managed by DVC.
        """
        # Run the 'dvc pull' command
        subprocess.run(["dvc", "pull"], check=True)

    @staticmethod
    def commit_and_push_dvc():
        """
        Commit and push changes to the DVC remote.

        This method uses the 'dvc commit' command to save changes to DVC files,
        and the 'dvc push' command to push any changes in the data or pipeline to the DVC remote.
        """
        # Commit changes to DVC files
        subprocess.run(["dvc", "commit"], check=True)

        # Push changes to the DVC remote
        subprocess.run(["dvc", "push"], check=True)
# if __name__ == "__main__":
#     dvc_instance = DVCUtils()
#     xxx = OSUtils()
#     print(xxx.check_file_exists('housing.py'))
#     # dvc_instance.add_remote_dvc(remote_name=dvc_instance.remote_name, remote_url=dvc_instance.remote_url)
#     # dvc_instance.pull_dvc()
