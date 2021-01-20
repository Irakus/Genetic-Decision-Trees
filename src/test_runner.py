from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from genetic_tree_constructor import GeneticDecisionTreeGenerator
from sklearn.tree import DecisionTreeClassifier
import sys
import timeit
import functools

def timed_test(name = 'Test'):
    def timed_decorator(func):
        @functools.wraps(func)
        def timer_wrapper(*args, **kwargs):
            start = timeit.timeit()
            value = func(*args, **kwargs)
            print("{} executed in ".format(name), timeit.timeit() - start)
            return value
        return timer_wrapper
    return timed_decorator

@timed_test(name = "Breast cancer test")
def run_bc(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress = 10):
    data = datasets.load_breast_cancer()
    return run_test('Breast cancer',data,file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress,has_features=True)

@timed_test(name = "Iris test")
def run_iris(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress = 10):
    data = datasets.load_iris()
    return run_test('Iris',data,file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress,has_features=True)

@timed_test(name = "Wine test")
def run_wine(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress = 10):
    data = datasets.load_wine()
    return run_test('Wine',data,file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress,has_features=True)

@timed_test(name = "Digits test")
def run_digits(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress = 10):
    data = datasets.load_digits()
    return run_test('Digits', data, file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress, has_features=False)


def run_digits_cut_data(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress = 10):
    data = datasets.load_digits()

    size_before = data.data.size
    nonzeros_before = np.count_nonzero(data.data)
    data.data = np.delete(data.data, np.arange(0, data.data.shape[0], 8), axis=1)
    data.data = np.delete(data.data, np.arange(6, data.data.shape[0], 7), axis=1)

    return run_test('Digits', data, file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress, has_features=False)

def run_digits_cut_targets(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress=10 ):
    data = datasets.load_digits()

    data.data, data.target = data.data[data.target <= 3], data.target[data.target <= 3]

    return run_test('Digits',  data, file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth,stop_after_no_progress, has_features=False)

def run_digits_cut_data_and_target(file=None,population_size=20,n_epochs=200, n_iterations=10, starting_population_min_depth=3,starting_population_max_depth=20,stop_after_no_progress=10):
    data = datasets.load_digits()
    data.data = np.delete(data.data, np.arange(0, data.data.shape[0], 8), axis=1)
    data.data = np.delete(data.data, np.arange(6, data.data.shape[0], 7), axis=1)
    data.data, data.target = data.data[data.target <= 3], data.target[data.target <= 3]

    return run_test('Digits', data, file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth, stop_after_no_progress, has_features=False)

def run_test(name, data,file,population_size,n_epochs, n_iterations, starting_population_min_depth,starting_population_max_depth, stop_after_no_progress,has_features = False):
    X, Y = data.data, data.target
    global X_test,Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=32,train_size=0.8)
    generator = GeneticDecisionTreeGenerator(population_size = population_size, 
                                            n_epochs = n_epochs,
                                            n_iterations = n_iterations,
                                            starting_population_min_depth = starting_population_min_depth, 
                                            starting_population_max_depth = starting_population_max_depth)
    trees = generator.fit(X_train,Y_train,stop_after_no_progress=10)
    best_tree = choose_best_tree(trees)

    Y_pred_train = np.apply_along_axis( best_tree.pred , axis=1, arr=X_train)
    Y_pred_test = np.apply_along_axis( best_tree.pred , axis=1, arr=X_test)
    print('{}: \n\ttrain accuarcy: {}\n\ttest accuarcy: {}'.format(name,accuracy_score(Y_train,Y_pred_train), accuracy_score(Y_test,Y_pred_test)))
    if file is not None:
        file.write(name + ': \n\ttrain accuracy: {}\n\taccuarcy error: {}\n'.format(accuracy_score(Y_train,Y_pred_train), accuracy_score(Y_test,Y_pred_test)))
        if has_features:
            file.write(best_tree.__str__(params_dictionary=data.feature_names,classes_dictionary=data.target_names))
        else:
            file.write(best_tree.__str__(classes_dictionary=None))
    try:
        result_data = [best_tree, data.feature_names, data.target_names]
        return result_data
    except KeyError as e:
        return best_tree, None, data.target_names
    except AttributeError as e:
        return best_tree, None, data.target_names

def choose_best_tree(trees):
    trees.sort(key=calc_test_error,reverse=True,)
    for tree in trees:
        print('Tree \n  test score: {}\n  train score: {}'.format(*get_both_scores(tree)))
    return trees[0]

def calc_test_error(tree):
    Y_pred = np.apply_along_axis( tree.pred , axis=1, arr=X_test)
    return accuracy_score(Y_test,Y_pred) + tree.get_score()

def get_both_scores(tree):
    Y_pred = np.apply_along_axis( tree.pred , axis=1, arr=X_test)
    return accuracy_score(Y_test,Y_pred) , tree.get_score()
