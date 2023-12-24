import math
from random import randint
import numpy
import numpy as np
import numpy as np
import scipy.special
import matplotlib.pyplot as plt


# Activation function for output layer
def activation_function(x):
    """
    Applies the sigmoid activation function to the input.

    Parameters:
    x (float): The input value.

    Returns:
    float: The output value after applying the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))  # or scipy.special.expit(x)


# neural network class definition
class neuralNetwork:
    """
    A class representing a neural network.

    Parameters:
    - layers_number (list): A list containing the number of nodes in each layer of the neural network.
    - initial_weight (list): A list containing the initial weights for the connections between the layers.
    - alpha (float): The learning rate of the neural network.
    - epoch (int): The number of training iterations.
    - batch_testing (int, optional): The number of samples to test in each batch. Defaults to 1.

    Attributes:
    - input_node_number (int): The number of nodes in the input layer.
    - hidden_node_number (int): The number of nodes in the hidden layer.
    - output_node_number (int): The number of nodes in the output layer.
    - iteration (int): The number of training iterations.
    - lr (float): The learning rate of the neural network.
    - batch_testing (int): The number of samples to test in each batch.
    - weight_input_hidden (ndarray): The weights for the connections between the input and hidden layers.
    - weight_hidden_output (ndarray): The weights for the connections between the hidden and output layers.
    - activation_function_sigmoid (function): The activation function used in the neural network.
    - error_by_training (list): A list to store the training errors during training.

    Methods:
    - train(dataset): Train the neural network using the given dataset.
    - testing(dataset): Test the neural network using the given dataset.
    - query(inputs_list): Query the neural network with the given inputs.
    - calculate_accuracy(dataset): Calculate the accuracy of the neural network using the given dataset.
    """

    def __init__(self, layers_number, initial_weight, alpha, epoch, batch_testing=1):
        self.input_node_number = layers_number[0]
        self.hidden_node_number = layers_number[1]
        self.output_node_number = layers_number[2]
        self.iteration = epoch
        self.lr = alpha

        self.batch_testing = batch_testing

        self.weight_input_hidden = numpy.array(initial_weight[0], dtype=float)
        self.weight_hidden_output = numpy.array(initial_weight[1], dtype=float)

        self.activation_function_sigmoid = lambda x: activation_function(x)
        # self.activation_function_sigmoid = lambda x: step_function(x)
        self.error_by_training = []

    def train(self, dataset):
        """
        Train the neural network using the given dataset.

        Parameters:
        - dataset (list): A list of tuples containing the input samples and their corresponding labels.

        Returns:
        None
        """
        self.error_by_training = []

        count = 0
        for epoch in range(self.iteration):
            dataset_length = len(dataset)
            selected_samples = dataset[randint(0, dataset_length - 1)]

            samples = selected_samples[0]
            label = selected_samples[1]

            inputs = numpy.array(samples, ndmin=2).T
            targets = numpy.array(label, ndmin=2).T

            hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
            hidden_outputs = self.activation_function_sigmoid(hidden_inputs)

            final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
            final_outputs = self.activation_function_sigmoid(final_inputs)

            # Backward
            output_errors = targets - final_outputs
            hidden_errors = numpy.dot(self.weight_hidden_output.T, output_errors)

            self.weight_hidden_output += self.lr * numpy.dot(
                (output_errors),
                numpy.transpose(hidden_outputs),
            )

            self.weight_input_hidden += self.lr * numpy.dot(
                (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                numpy.transpose(inputs),
            )

            count += 1
            if count > self.batch_testing:
                count = count % self.batch_testing
                self.error_by_training.append(self.testing(dataset))

            accuracy = self.calculate_accuracy(dataset)
            if accuracy == 100.0:
                print(f"Converged after {epoch + 1} iterations")
                break

    def testing(self, dataset):
        """
        Test the neural network using the given dataset.

        Parameters:
        - dataset (list): A list of tuples containing the input samples and their corresponding labels.

        Returns:
        float: The total error of the neural network on the given dataset.
        """
        result = []
        for sample, label in dataset:
            prediction = self.query(sample)
            error = label - [prediction[0][0], prediction[1][0]]
            result.append(sum(numpy.absolute(error)))
        return sum(result)

    def query(self, inputs_list):
        """
        Query the neural network with the given inputs.

        Parameters:
        - inputs_list (list): A list of input values.

        Returns:
        ndarray: The output values of the neural network.
        """
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weight_input_hidden, inputs)
        hidden_outputs = self.activation_function_sigmoid(hidden_inputs)

        final_inputs = numpy.dot(self.weight_hidden_output, hidden_outputs)
        final_outputs = self.activation_function_sigmoid(final_inputs)
        final_outputs = np.round(final_outputs, 0)

        return final_outputs

    def calculate_accuracy(self, dataset):
        """
        Calculate the accuracy of the neural network using the given dataset.

        Parameters:
        - dataset (list): A list of tuples containing the input samples and their corresponding labels.

        Returns:
        float: The accuracy of the neural network on the given dataset.
        """
        correct_predictions = 0
        total_samples = len(dataset)

        for sample, label in dataset:
            prediction = self.query(sample)
            rounded_prediction = np.round(prediction).astype(int).flatten()
            if np.array_equal(rounded_prediction, label.astype(int)):
                correct_predictions += 1

        accuracy = (correct_predictions / total_samples) * 100
        return accuracy


####### init parameters ################################
# learning rate is 0.01
learning_rate = 0.01
iteration = 20000

# number of input, hidden and output nodes
input_nodes = 2
hidden_nodes = 3
output_nodes = 2
layers = [input_nodes, hidden_nodes, output_nodes]

# initial weight
initial_weight = [
    [[0.1, -0.2], [0, 0.2], [0.3, -0.4]],
    [[-0.4, 0.1, 0.6], [0.2, -0.1, -0.2]],
]

dataset = numpy.array(
    [
        [[0.6, 0.1], [1, 0]],
        [[0.2, 0.3], [0, 1]],
        [[0.4, 0.4], [0, 1]],
        [[0.4, 0.2], [0, 1]],
        [[0.5, 0.3], [1, 0]],
        [[0.1, 0.2], [0, 1]],
        [[0.8, 0.7], [1, 0]],
    ],
    dtype=float,
)
prediction_dataset = numpy.array(
    [
        [0.3, 0.9],
        [0.7, 0.8],
        [0.6, 0.8],
        [0.2, 0.1],
    ],
)

# init model
neurel_network = neuralNetwork(
    layers, initial_weight, learning_rate, iteration, batch_testing=50
)
# testing our model
neurel_network.testing(dataset)


# show models performance before training
print("Models performance before training")
print("Samples Value\t", "\t Pred [Clou, Vis]", "\t [Clou, Vis]")
for sample in dataset:
    result = list(neurel_network.query(sample))
    print(
        "Sample:",
        [sample[0][0], sample[0][1]],
        " \t Prediction",
        [result[0][0], result[1][0]],
        "\t Label:",
        sample[1],
    )
print("\nNumber of errors")
print(neurel_network.testing(dataset))


# train our model
print("\nModels performance after training")
neurel_network.train(dataset)
print("Samples Value\t", "\t Pred [Clou, Vis]", "\t [Clou, Vis]")
for sample in dataset:
    result = list(neurel_network.query(sample))
    print(
        "Sample:",
        [sample[0][0], sample[0][1]],
        " \t Prediction",
        [result[0][0], result[1][0]],
        "\t Label:",
        sample[1],
    )
print("\nNumber of errors")
print(neurel_network.testing(dataset))


# After training, calculate and print accuracy
accuracy = neurel_network.calculate_accuracy(dataset)
print(f"Accuracy after training: {accuracy:.2f}%")

# show final weight
print("\nWeight Input Hidden:\n", neurel_network.weight_input_hidden)
print("Weight Hidden Output:\n", neurel_network.weight_hidden_output)

# test our unlabeled data
print("\n")
print("Test our unlabeled data")
print("Samples Value\t", "\t Prediction [Clou, Vis]")
for sample in prediction_dataset:
    result = list(neurel_network.query(sample))
    print(
        "Sample", [sample[0], sample[1]], "\t Prediction", [result[0][0], result[1][0]]
    )

# show error rate
length_error = len(neurel_network.error_by_training)
iterations = np.array(range(length_error)) * neurel_network.batch_testing
fig, ax = plt.subplots()
ax.plot(iterations, neurel_network.error_by_training)
ax.grid(True, linestyle="-.")
ax.tick_params(labelcolor="r", labelsize="medium", width=3)
plt.show()


# plot data after training and prediction
my_dataset = numpy.array(
    [
        # labled
        [[0.6, 0.1], [1, 0]],
        [[0.2, 0.3], [0, 1]],
        [[0.4, 0.4], [0, 1]],
        [[0.4, 0.2], [0, 1]],
        [[0.5, 0.3], [1, 0]],
        [[0.1, 0.2], [0, 1]],
        [[0.8, 0.7], [1, 0]],
        # unlabeled
        [[0.3, 0.9], [0, 1]],
        [[0.7, 0.8], [1, 0]],
        [[0.6, 0.8], [1, 0]],
        [[0.2, 0.1], [0, 1]],
    ],
)

get_x = lambda x: [i[0] for i in x]
get_y = lambda x: [i[1] for i in x]

area1, area2, area3 = [], [], []

for sample, label in my_dataset:
    if label[0] == 1:
        area1.append(sample)
    elif label[0] == 0:
        area2.append(sample)
    else:
        area3.append(sample)

plt.scatter(get_x(area1), get_y(area1), label="Clou", marker="o")
plt.scatter(get_x(area2), get_y(area2), label="Vis", marker="+")
plt.scatter(get_x(area3), get_y(area3), label="none", marker="*")

plt.legend(["Clou", "Vis", "none"], loc="upper left")


plt.show()
