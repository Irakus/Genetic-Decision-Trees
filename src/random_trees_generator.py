from random import random
from random import choices
from decision_tree import NodesTypes,Criteria,Decision,Leaf

def generate_random_tree(min_depth, max_depth, classes_count, record_data_count, record_data_minmax):
    if min_depth <= 0:
        raise ValueError('Trees must have depth of at least 1')
    else:
        type = choices(list(NodesTypes))[0]
        if type is NodesTypes.DECISION:
            criteria = Criteria(record_data_count,record_data_minmax)
            root = Decision(criteria)
            root.add_child(tree_node(1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
            root.add_child(tree_node(1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
        elif type is NodesTypes.LEAF:
            root = Leaf(classes_count)
        else:
            raise ValueError('Wrong enum type: {}'.format(type))
    return root


def tree_node(level, min_depth, max_depth, classes_count, record_data_count, record_data_minmax):
    node = None
    if level == max_depth:
        node = Leaf(classes_count)
    elif level < min_depth:
        criteria = Criteria(record_data_count, record_data_minmax)
        node = Decision(criteria)
        node.add_child(tree_node(level + 1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
        node.add_child(tree_node(level + 1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
    elif min_depth <= level < max_depth:
        type = choices(list(NodesTypes))[0]
        if type == NodesTypes.DECISION:
            criteria = Criteria(record_data_count,record_data_minmax)
            node = Decision(criteria)
            node.add_child(tree_node(level + 1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
            node.add_child(tree_node(level + 1, min_depth, max_depth, classes_count, record_data_count, record_data_minmax))
        elif type == NodesTypes.LEAF:
            node = Leaf(classes_count)
        else:
            raise ValueError('Invalid node type: {}'.format(type))
    else:
        raise ValueError('Invalid level value: {}. Depth should be betweeen {} and {}'.format(level, min_depth, max_depth))
    return node