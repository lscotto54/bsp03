from src.fitness.base_ff_classes.base_ff import base_ff
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class cifar10_cnnc(base_ff):
    """
    Py-max is a max-style problem where the goal is to generate a function
    which outputs a large number. In the standard GP Max [Gathercole and
    Ross] problem this function can only use the constant (0.5) and functions
    (+, *). The Py-max problem allows more programming: numerical expressions,
    assignment statements and loops. See pymax.pybnf.

    Chris Gathercole and Peter Ross. An adverse interaction between crossover
    and restricted tree depth in genetic programming. In John R. Koza,
    David E. Goldberg, David B. Fogel, and Rick L. Riolo, editors, Genetic
    Programming 1996: Proceedings of the First Annual Conference.
    """

    maximise = True  # True as it ever was.

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

        self.num_classes = 10
        self.epochs = 50

        # The data, split between train and test sets:
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Convert class vectors to binary class matrices.
        self.y_train = keras.utils.to_categorical(self.y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, self.num_classes)

        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255



    def evaluate(self, ind, **kwargs):
        # ind.phenotype will be a string, including function definitions etc.
        # When we exec it, it will create a value XXX_output_XXX, but we exec
        # inside an empty dict for safety.
        p, d = ind.phenotype, {"x_train": self.x_train, "y_train": self.y_train, "x_test": self.x_test, "y_test": self.y_test, "num_classes": self.num_classes, "epochs": self.epochs}
        print(ind.phenotype)

        # Exec the phenotype.
        exec(p, d)

        # Get the output
        # s = d['XXX_output_XXX']  # this is the program's output: a number.
        score = d['XXX_output_XXX']
        print("Test loss is ", score[0])
        print("Test accuracy is ", score[1])

        return score[1]
