# Paraphrase Detection

This repository contains code for paraphrase detection using various machine learning models. The task involves determining whether two given sentences convey the same meaning or not.

## Table of Contents

- [Dataset](#dataset)
- [Setup](#setup)
- [Models](#models)
    - [Logistic Regression](#logistic-regression)
    - [Siamese Neural Network](#siamese-neural-network)
    - [DistilBERT](#distilbert)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#discussion)

## Dataset

The dataset used is the Quora Question Pairs dataset, which contains pairs of questions labeled as either paraphrases or non-paraphrases. The dataset is split into train, validation, and test sets.

## Setup

1. Clone the repository: `git clone https://github.com/shreyas1104/Paraphrase-Detection-using-Siamese-NN-and-DistilBERT.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Download the dataset from `https://huggingface.co/datasets/paws`

## Models

The following models are implemented for paraphrase detection:

### Logistic Regression

A traditional machine learning model using TF-IDF vectorization and logistic regression.

### Siamese Neural Network

A neural network architecture with shared weights for encoding sentence pairs, using LSTM and an embedding layer.

### DistilBERT

A transformer-based model fine-tuned on the paraphrase detection task using the DistilBERT pre-trained model.

## Usage

1. Run the respective code for each model to train and evaluate it on the dataset.
2. The trained models can be saved and loaded for future use.
3. Modify the code as needed to experiment with different hyperparameters or preprocess the data differently.

## Results

The performance of each model on the test set is reported as accuracy:

- Logistic Regression: 0.5119
- Siamese Neural Network: 0.6105
- DistilBERT: 0.8857

## Discussion

From the results, we can observe a significant performance gap between the traditional machine learning model (Logistic Regression) and the deep learning models (Siamese Neural Network and DistilBERT). This highlights the effectiveness of neural network architectures in capturing the semantic nuances and contextual information required for the paraphrase detection task.

The Logistic Regression model, which relies on TF-IDF vectorization, achieves an accuracy of 0.5119, which is only slightly better than random guessing. This could be due to the inherent limitations of bag-of-words representations in capturing the underlying semantics of the sentences.

The Siamese Neural Network, which uses LSTM and word embeddings, performs better with an accuracy of 0.6105. By encoding the sequential information and leveraging pre-trained word embeddings, the model can capture richer contextual information than the Logistic Regression model.

The DistilBERT model, a transformer-based model fine-tuned on the paraphrase detection task, achieves the highest accuracy of 0.8857. This impressive performance can be attributed to the self-attention mechanism and the ability of transformer models to capture long-range dependencies and semantic relationships effectively.

Note: these results are specific to the dataset and preprocessing techniques used. Further improvements could potentially be achieved by exploring different architectures, hyperparameter tuning, and incorporating additional linguistic features or external knowledge sources.
