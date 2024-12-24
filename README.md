# Graph-Neural-Network-for-Image-Classification

This project implements a Graph Neural Network (GNN) for image classification on the MNIST dataset, based on the referenced research paper. The project leverages PyTorch Geometric to construct and train a GNN model with 5 GATConv layers. The results are compared to a benchmark 5-layer Convolutional Neural Network (CNN).

## Key Features

GNN Implementation: Built a GNN with 5 GATConv layers for effective image classification.

Image Representation: Transformed MNIST images into graph format for input into the GNN.

Benchmark Comparison: Achieved 99.08% test accuracy on a 5-layer CNN benchmark.

Hyperparameter Tuning: Optimized learning rate, dropout, and other hyperparameters to improve GNN performance.

Visualizations: Created graph-based visualizations of input images.

## Tech Stack

Frameworks and Libraries: PyTorch Geometric, PyTorch, Torchvision, Numpy, Matplotlib

Programming Language: Python

## Dataset

The MNIST dataset is used for training and testing. Each image in the dataset is converted into a graph representation, where nodes correspond to pixels and edges are based on pixel adjacency.

## Results

GNN Performance: Achieved ~ 97% test accuracy with the 5 GATConv layers.

CNN: Achieved 99.08% test accuracy.

Visual Analysis: Graph-based visualizations of MNIST images provided insights into model performance.

## File Structure

preprocess_mnist.py: Script to preprocess MNIST images into graph format.

train_gnn.py: Script to train the GNN model.

evaluate_gnn.py: Script to evaluate the GNN and generate visualizations.

models/: Contains model definitions for GNN and CNN.

data/: Directory for storing the MNIST dataset and processed graphs.

visualizations/: Directory for storing visualization outputs.

## Future Work

Extend to more complex datasets (e.g., CIFAR-10).

Explore different GNN architectures and layer types.

Investigate transfer learning approaches for GNNs.

References

Referenced research paper for GNN implementation. [Paper](https://iopscience.iop.org/article/10.1088/1742-6596/1871/1/012071/pdf)

Thank you for exploring this project! 
