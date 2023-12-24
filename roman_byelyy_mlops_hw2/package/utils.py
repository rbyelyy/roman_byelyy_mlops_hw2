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

    def add_remote_dvc(self, remote_name, remote_url):
        print(remote_name)
        subprocess.run(["dvc", "remote", "add", "-f", remote_name, remote_url], check=True)

    def pull_dvc(self):  # добавьте этот метод
        subprocess.run(["dvc", "pull"], check=True)

if __name__ == "__main__":
    dvc_instance = DVCUtils()
    xxx = OSUtils()
    print(xxx.check_file_exists('housing.py'))
    # dvc_instance.add_remote_dvc(remote_name=dvc_instance.remote_name, remote_url=dvc_instance.remote_url)
    # dvc_instance.pull_dvc()
