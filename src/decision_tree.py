from abc import ABCMeta, abstractmethod
from copy import deepcopy
from enum import Enum
import random

class NodesTypes(Enum):
    LEAF = 0
    DECISION = 1

class MutationTypes(Enum):
    CRITERIA_CHANGE = 0
    CHILDREN_SWAP = 1
    MUTATE_NEXT_NODE = 2

LEFT_CHILD = 0
RIGH_CHILD = 1

class Criteria:
    def __init__(self, record_data_count, record_data_minmax):
        self._index = random.randrange(record_data_count)
        self._treshold = random.uniform(record_data_minmax[self._index][0], record_data_minmax[self._index][1])
        self._record_data_count = record_data_count
        self._record_data_minmax = record_data_minmax

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        copied_criteria = Criteria(deepcopy(self._record_data_count), deepcopy(self._record_data_minmax))
        copied_criteria.set_parameters(deepcopy(self._index), deepcopy(self._treshold))
        return copied_criteria

    def reroll_criteria(self):
        self._index = random.randrange(self._record_data_count)
        self._treshold = random.uniform(self._record_data_minmax[self._index][0], self._record_data_minmax[self._index][1])

    def pass_criteria(self, tested):
        try:
            return tested[self._index] >= self._treshold
        except RecursionError:
            return 0

    def set_parameters(self, index,threshold,):
        self._index = index
        self._treshold = threshold

    def get_parameters(self):
        return self._index, self._treshold

class Node(metaclass = ABCMeta):
    @abstractmethod
    def pred(self):
        pass

    @abstractmethod
    def mutate(self):
        pass
    
    @abstractmethod
    def copy_branch(self):
        pass

class Decision(Node):
    def __init__(self, criteria):
        self._children = []
        self._criteria = criteria
        self._score = 0.0

    def __str__(self, level=0, params_dictionary=None, classes_dictionary=None):
        param_name_index, threshold = self._criteria.get_parameters() 
        ret = '{}{}>={}\n'.format('\t' * level, get_name_from_dict(param_name_index, params_dictionary), threshold)
        ret += self._children[LEFT_CHILD].__str__(level + 1, params_dictionary, classes_dictionary) + self._children[RIGH_CHILD].__str__(level + 1, params_dictionary, classes_dictionary)
        return ret

    def __deepcopy__(self, memodict={}):
        copied_decision = Decision(deepcopy(self._criteria))
        copied_decision.add_child(deepcopy(self._children[LEFT_CHILD]))
        copied_decision.add_child(deepcopy(self._children[RIGH_CHILD]))
        copied_decision.set_score(deepcopy(self._score))
        return copied_decision

    def add_child(self,new_child):
        self._children.append(new_child)

    def mutate(self):
        mutation_type = random.choices(list(MutationTypes), weights=[.1,.1,.8])[0]
        if mutation_type == MutationTypes.CRITERIA_CHANGE:
            self._criteria.reroll_criteria()
        elif mutation_type == MutationTypes.CHILDREN_SWAP:
            tmp_child = self._children[LEFT_CHILD]
            self._children[LEFT_CHILD] = self._children[RIGH_CHILD]
            self._children[RIGH_CHILD] = tmp_child
        elif mutation_type == MutationTypes.MUTATE_NEXT_NODE:
            self.__get_random_child().mutate()
        else:
            raise UnsupportedMutationType('Unknown mutation type {}'.format(mutation_type))

    def pred(self, data):
        if self._criteria.pass_criteria(data):
            return self._children[LEFT_CHILD].pred(data)
        else:
            return self._children[RIGH_CHILD].pred(data)

    def copy_branch(self):
        what_to_copy = random.random()
        if what_to_copy < 0.47:
            return self._children[LEFT_CHILD].copy_branch()
        elif what_to_copy < 0.95:
            return self._children[RIGH_CHILD].copy_branch()
        else:
            return deepcopy(self)

    def paste_branch(self, branch):
        where_to_pase = random.random()
        if isinstance(self._children[LEFT_CHILD], Leaf) and isinstance(self._children[RIGH_CHILD], Leaf):
            which_child = self.__get_random_child_index()
            self.__replace_child_with_branch(which_child, branch)
        elif isinstance(self._children[LEFT_CHILD], Leaf) and not isinstance(self._children[RIGH_CHILD], Leaf):
            self.__replace_child_with_branch(LEFT_CHILD, branch)
        elif not isinstance(self._children[LEFT_CHILD], Leaf) and isinstance(self._children[RIGH_CHILD], Leaf):
            self.__replace_child_with_branch(RIGH_CHILD, branch)
        else:
            if where_to_pase < 0.47:
                self._children[LEFT_CHILD].paste_branch(branch)
            elif where_to_pase < 0.95:
                self._children[RIGH_CHILD].paste_branch(branch)
            else:
                which_child = self.__get_random_child_index()
                self._children[which_child] = branch

    def set_score(self,score):
        self._score = score

    def get_score(self):
        return self._score

    def __replace_child_with_branch(self, child_index, branch):
        self._children[child_index] = branch

    def __get_random_child_index(self):
        return random.choice([LEFT_CHILD, RIGH_CHILD])

    def __get_random_child(self):
        return self._children[self.__get_random_child_index()]

    
class Leaf(Node):
    def __init__(self,classes_count):
        self._result = random.randrange(classes_count)
        self._classes_count = classes_count
        self._score = 0.0

    def __deepcopy__(self, memodict={}):
        copied_leaf = Leaf(deepcopy(self._classes_count))
        copied_leaf.set_result(deepcopy(self._result))
        copied_leaf.set_score(deepcopy(self._score))
        return copied_leaf

    def __str__(self,level=0,dictionary=None,classes_dictionary=None): 
        return '{} Class: {}\n'.format('\t' * level, get_name_from_dict(self._result,classes_dictionary))

    def pred(self,  data):
        return self._result

    def mutate(self):
        self._result = random.randrange(self._classes_count)

    def set_result(self, result):
        self._result = result

    def copy_branch(self):
        return deepcopy(self)

    def set_score(self, score):
        self._score = score

    def get_score(self):
        return self._score    

class UnsupportedMutationType(Exception):
    def __init__(self,message):
        super().__init__(message)

def get_name_from_dict(index, dict = None):
    if dict is None:
        return index
    else:
        return dict[index]