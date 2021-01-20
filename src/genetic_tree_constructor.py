from copy import deepcopy

from random_trees_generator import generate_random_tree, Leaf
import numpy as np
from sklearn.metrics import accuracy_score
import random
import math

SCORING_FACTOR_RATE = 1
SURVIVAL_RATE = 0.2
MUTATIONS_PER_CYCLE = 5


class GeneticDecisionTreeGenerator():

    def __init__(self,population_size = 20, n_epochs = 20, n_iterations=1, stop_rate=1.0, starting_population_min_depth = 5, starting_population_max_depth=15):
        self._population_size = population_size
        self._n_epochs = n_epochs
        self._n_iterations = n_iterations
        self._stop_rate = stop_rate
        self._X_train = None
        self._Y_train = None
        self.X_minmax = None
        self._classes_count = None
        self._record_data_count = None
        self._starting_population_min_depth = starting_population_min_depth
        self._starting_population_max_depth = starting_population_max_depth

    def fit(self, X_train, Y_train, log_info=True, stop_after_no_progress=100):
        self._X_train = X_train
        self._Y_train = Y_train
        X_min = X_train.min(axis=0)
        X_min = X_min.reshape(len(X_min), 1)
        X_max = X_train.max(axis=0)
        X_max = X_max.reshape(len(X_max), 1)
        self._X_minmax = np.concatenate((X_min, X_max), axis=1)
        self._classes_count = len(np.unique(Y_train))
        self._record_data_count = X_train.shape[1]
        best_trees = []
        for iteration in range(self._n_iterations):
            population = self.__generate_starting_population()
            iteration_best = self.__find_best(population)
            if log_info:
                print('Iteration {}, Starting population best score on train set: {}'.format(iteration,self.__get_tree_score(iteration_best) ** (1/SCORING_FACTOR_RATE)))
            epochs_without_progress = 0
            for epoch in range(self._n_epochs):
                population = self._run_epoch(population)
                epoch_best = self.__find_best(population)
                if self.__compare_trees(epoch_best, iteration_best):
                    iteration_best = epoch_best
                    epochs_without_progress = 0
                else:
                    epochs_without_progress += 1
                    if epochs_without_progress >= stop_after_no_progress:
                        break
                if log_info:
                    print('Iteration {}, Epoch {} best score on train set: {}'.format(iteration, epoch, self.__get_tree_score(epoch_best) ** (1 / SCORING_FACTOR_RATE)))
            best_trees.append(iteration_best)
        return best_trees



    def __generate_starting_population(self):
        population = []
        for i in range(self._population_size):
            new_tree = generate_random_tree(self._starting_population_min_depth, self._starting_population_max_depth,
                                 self._classes_count, self._record_data_count, self._X_minmax)
            self.__evaluate_tree(new_tree)
            population.append(new_tree)
        return population

    def __evaluate_tree(self, tree):
        Y_pred = np.apply_along_axis(tree.pred, axis=1, arr=self._X_train)
        tree.set_score(accuracy_score(self._Y_train,Y_pred) ** SCORING_FACTOR_RATE)

    def _run_epoch(self, population):
        population.sort(key=self.__get_tree_score, reverse=True)
        accuracy_sum = 0
        for pop in range(math.ceil(len(population) * SURVIVAL_RATE)):
            accuracy_sum += self.__get_tree_score(population[pop])
        new_population = []
        surviving_trees = math.ceil(len(population) * SURVIVAL_RATE)
        for surviving_tree_idx in range(surviving_trees): #selection
            new_population.append(deepcopy(population[surviving_tree_idx]))
            new_population.append(deepcopy(population[surviving_tree_idx]))
        for treeind in range(len(population) - 2 * surviving_trees):
            op = random.random()
            if op <= 0.4: #crossing
                tree1 = deepcopy(self.__select_with_probability(population,accuracy_sum))
                tree2 = deepcopy(self.__select_with_probability(population,accuracy_sum))
                if type(tree1) is Leaf:
                    n_leaf_selection = 0
                    while type(tree1) is Leaf and n_leaf_selection < len(population) * 5:
                        n_leaf_selection += 1
                        tree1 = deepcopy(self.__select_with_probability(population,accuracy_sum))
                    if type(tree1) is Leaf:
                        new_population.append(tree1)
                        continue
                tree1.paste_branch(tree2.copy_branch())
                self.__evaluate_tree(tree1)
                new_population.append(tree1)
            else: #mutation
                tree1 = deepcopy(self.__select_with_probability(population, accuracy_sum))
                for i in range(random.randrange(MUTATIONS_PER_CYCLE)):
                    tree1.mutate()
                self.__evaluate_tree(tree1)
                new_population.append(tree1)
        return new_population

    def __compare_trees(self, tree1, tree2):
        return self.__get_tree_score(tree1) < self.__get_tree_score(tree2)

    def __find_best(self, population):
        population.sort(key=self.__get_tree_score, reverse=True)
        return population[0]

    def __select_with_probability(self,population, accuracy_sum):
        selected = random.uniform(0.0, accuracy_sum)
        selected_index = 0
        current_value = self.__get_tree_score(population[0])
        while selected > current_value and selected_index < len(population):
            selected_index += 1
            current_value += self.__get_tree_score(population[selected_index])
        return population[selected_index]

    def __get_tree_score(self,tree):
        return tree.get_score()