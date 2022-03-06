import sys
import pennylane as qml
from pennylane import numpy as np

DATA_SIZE = 250

def square_loss(labels, predictions):
    """Computes the standard square loss between model predictions and true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - loss (float): the square loss
    """

    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss


def accuracy(labels, predictions):
    """Computes the accuracy of the model's predictions against the true labels.

    Args:
        - labels (list(int)): True labels (1/-1 for the ordered/disordered phases)
        - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)

    Returns:
        - acc (float): The accuracy.
    """

    acc = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            acc = acc + 1
    acc = acc / len(labels)

    return acc


def classify_ising_data(ising_configs, labels):
    """Learn the phases of the classical Ising model.

    Args:
        - ising_configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
        - labels (np.ndarray): 250 rows of labels (1 or -1)

    Returns:
        - predictions (list(int)): Your final model predictions
    """

    num_wires = ising_configs.shape[1] 
    dev = qml.device("default.qubit", wires=num_wires) 

    @qml.qnode(dev)
    def circuit(config, params):
        """Design the variational circuit.

        Args:
            - config (np.ndarray): an Ising model configuration 
            - params (np.ndarray): Parameters for the variational circuit

        Returns:
            - probability (list(float)): Probability of different states
        """
        for i in range(num_wires):
            if config[i]==1:
                qml.PauliX(wires=i)
            qml.RY(params[i],wires=i)
        return qml.probs(wires=range(num_wires))

    def predict(configs, params):
        """Returns the prediction for the different configurations.  

        Args:
            - configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations 
            - params (np.ndarray): Parameters for the variational circuit

        Returns:
            - predictions (list(int)): Model predictions (1/-1 for the ordered/disordered phases)
        """
        predictions = []
        for i in range(len(configs)):
            result = circuit(configs[i],params)
            result = result[0] + result[2**num_wires-1]
            if result==1:
                predictions.append(1)
            else:
                predictions.append(-1)
        return np.array(predictions, requires_grad=False)

    def cost(params,configs, Y):
        """Defines the cost function of the circuit.  

        Args: 
            - params (np.ndarray): Parameters for the variational circuit
            - configs (np.ndarray): 250 rows of binary (0 and 1) Ising model configurations
            - Y (np.ndarray): 250 rows of labels (1 or -1)

        Returns:
            - float: Square loss between true labels and predicted labels
        """
        predictions = predict(configs, params)
        return square_loss(Y, predictions)

    params = np.array([np.random.random()*np.pi for i in range(num_wires)], requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=0.8)
    steps = 250

    for step in range(steps):
        params, ising_configs, labels  = opt.step(cost, params, ising_configs, labels)

    predictions = predict(ising_configs,params)

    return predictions


if __name__ == "__main__":
    inputs = np.array(
        sys.stdin.read().split(","), dtype=int, requires_grad=False
    ).reshape(DATA_SIZE, -1)
    ising_configs = inputs[:, :-1]
    labels = inputs[:, -1]
    predictions = classify_ising_data(ising_configs, labels)
    print(*predictions, sep=",")
