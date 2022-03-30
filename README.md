# Genetic Algorithm for generating Decision Trees
## Tree structure
Trees are represented by classes `Leaf` and `Decision`. Both of these classes have `pred(data)` method which helps in classifying incoming data.
Leafs represent classes that object is put into.
Decision are a bit more complex. They have 2 children(of type `Leaf` or `Decision`) and during `pred()` method they use criteria to decide, to which of their children they want to pass decision next.
##  Generation (starting population)
Individual trees are generated using `generate_random_tree()`. `GeneticTreeContstuctor` class uses this method over and over to create starting population of trees. They are created completely randomly.
## Selection
Selection is based on number of test cases that are classified correctly. Only 20% of fittest trees are used in the next steps of iteration.
## New population
In addition to those 20%, rest of trees of the next population come from crossing(around 40% of new trees) and mutation(~60%) of the fittest trees.
### Mutation
Mutation can take 3 forms:
* Changing places of children in random `Decision` node
* Changing `Criteria` of random `Decision` node
* Changing class of random `Leaf`
### Crossing
Crossing is also very simple. It randomly picks subtree of one tree and pastes that in random place of another tree.
## Termination
Algorithm stops when last iteration is over, or if for previously set number of iterations, algorithm cannot generate new fittest tree.