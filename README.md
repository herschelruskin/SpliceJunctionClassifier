# Splice Junction DNA Classification

This project implements machine learning models to classify DNA sequences containing splice junctions. The task is to determine whether a sequence represents an exon–intron boundary, intron–exon boundary, or neither.

## Dataset

The dataset consists of:

- 3,190 labeled DNA sequences
- Each sequence represents a fixed-length nucleotide window around potential splice junctions

Splice junction dataset originally from the UCI Machine Learning Repository.

https://archive.ics.uci.edu/ml/datasets/splice+junction+gene+sequences

## Feature Engineering

DNA sequences were converted into numerical features using one-hot encoding of nucleotides (A, C, G, T).  
This resulted in 287 input features representing the encoded sequence.

## Models Evaluated

Several classification algorithms were implemented and compared:

- Support Vector Machine (SVM)
- Multi-Layer Perceptron (Neural Network)
- Decision Tree
- K-Nearest Neighbors
- Naïve Bayes

Hyperparameters were tuned using **GridSearchCV with 5-fold cross-validation**.

## Results

The best performance was achieved with an **SVM classifier**, reaching approximately:

- ~96% accuracy
- ~0.95 macro F1 score

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn

## Purpose

This project demonstrates basic bioinformatics feature encoding and supervised classification techniques applied to genomic sequence data.

## Installation

pip install -r requirements.txt
