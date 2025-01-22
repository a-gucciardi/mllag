Here's a simple diagram representing the neural network architecture used in the previous code:
```markdown
+-----------------------+
|         Input         |
|  (2 features: X1, X2)  |
+-----------------------+
            |
            | 64 neurons with ReLU activation
            v
+-------------------------------+
|          Hidden Layer 1       |
|  (Dense, 64 units, ReLU)      |
+-------------------------------+
            |
            | 32 neurons with ReLU activation
            v
+-------------------------------+
|          Hidden Layer 2       |
|  (Dense, 32 units, ReLU)      |
+-------------------------------+
            |
            | 1 neuron with sigmoid activation
            v
+-------------------------------+
|         Output Layer        |
|  (Dense, 1 unit, Sigmoid)    |
|  (Binary Classification)     |
+-------------------------------+
```
This diagram illustrates the architecture of the model:

*   The **Input Layer** receives the input data with two features (`X1` and `X2`).
*   The first **Hidden Layer** consists of 64 neurons, each applying a ReLU activation function to its output. This layer processes the input data.
*   The second **Hidden Layer** has 32 neurons, also using ReLU activation. It further refines the features extracted by the previous layer.
*   Finally, the **Output Layer** contains one neuron with a sigmoid activation function, which is suitable for binary classification tasks.

Here's an equivalent representation in a more formal diagram format (inspired by Keras' model diagrams):
```markdown
Input (2) --> Dense (64, ReLU) --> Dense (32, ReLU) --> Dense (1, Sigmoid)
```
This notation represents the neural network architecture as follows:

*   `Input (2)`: The input layer with 2 features.
*   `Dense (64, ReLU)`: A fully connected (dense) layer with 64 units and ReLU activation.
*   `Dense (32, ReLU)`: Another dense layer with 32 units and ReLU activation.
*   `Dense (1, Sigmoid)`: The output layer with 1 unit and sigmoid activation.

Note: In this simplified diagram, the bias terms and weights are not explicitly represented.