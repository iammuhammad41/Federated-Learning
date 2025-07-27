# Federated MNIST Classification with PyTorch

This project demonstrates a simple federated learning setup on the MNIST dataset using PyTorch. Multiple clients independently train a small convolutional neural network on disjoint subsets of the data. Model parameters are averaged each communication round, and the global model is evaluated on a held-out development set.



## 📁 Repository Structure

```
fed-mnist/
├── federated_mnist.ipynb   # Notebook containing full code and results
└── README.md               # This file
```



## ⚙️ Requirements

* Python 3.6+
* PyTorch
* torchvision
* matplotlib

Install dependencies via pip:

```bash
pip install torch torchvision matplotlib
```



## 🔧 Data Preparation

1. **Download MNIST**: The code uses `torchvision.datasets.MNIST` to download and load the data into `/content/input`.
2. **Train/Dev/Test Splits**:

   * Train set: 83% of the original training data (49,800 examples)
   * Dev set: 17% of the original training data (10,200 examples)
   * Test set: 10,000 examples



## 🧠 Model Definition

* **`FederatedNet`**: A small CNN with two `Conv2d` layers, ReLU activations, max pooling, and a single linear output layer.
* **Parameter Management**: Methods to extract (`get_parameters`) and apply (`apply_parameters`) layer weights and biases for federated averaging.



## 🚀 Federated Training Flow

1. **Clients**: Split the training data evenly among `num_clients` (default 8). Each client wraps its subset in a `Client` object.
2. **Local Training**: In each round, clients receive the current global parameters, train locally for `epochs_per_client` epochs, and return updated parameters.
3. **Server Aggregation**: The global model averages client parameters weighted by dataset size.
4. **Evaluation**: After each round, the global model is evaluated on train and dev sets.
5. **Rounds**: Repeat for `rounds` communication rounds (default 30).



## 🔍 Hyperparameters

| Parameter           | Default | Description                            |
| ------------------- | ------- | -------------------------------------- |
| `num_clients`       | 8       | Number of federated clients            |
| `rounds`            | 30      | Communication rounds                   |
| `batch_size`        | 128     | Mini-batch size for local training     |
| `epochs_per_client` | 3       | Local training epochs per round        |
| `learning_rate`     | 0.02    | SGD learning rate for local optimizers |



## 📈 Results

* The notebook plots training and dev loss across rounds, showing improved dev accuracy (e.g., \~0.98 over 30 rounds).
* Detailed per-client losses and accuracies are printed each round.



## ▶️ Usage

1. Open `federated_mnist.ipynb` in Jupyter or Google Colab.
2. Run all cells in order to:

   * Download data
   * Initialize clients and global model
   * Perform federated training
   * Plot results
