import random
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.tree as TreeTools
import numpy as np
from random_trees_generator import generate_random_tree
from test_runner import run_digits, run_iris, run_bc, run_wine, run_digits_cut_targets, run_digits_cut_data_and_target, run_digits_cut_data


if __name__ == '__main__':
    with open('tmp.txt','w+') as file:
        run_bc(file)
        run_iris(file)
        run_wine(file)




