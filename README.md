# Language Model Quantization and Tokenization

This repository contains a Jupyter Notebook that demonstrates two key tasks related to natural language processing (NLP):

## 1. Replicating a Simple Tokenizer inspired by Andrej Karpathy's implementation.
## 2. Converting a Language Model to int8 using the Hugging Face Optimum library and ONNX Runtime for efficient inference.

# Table of Contents

## Introduction
## Prerequisites
## Installation
## Usage
## Notebook Sections
###  Section 1: Replicating the Andrej Karpathy Tokenizer Code
###  Section 2: Convert a Language Model to int8 Using Optimum
## Contributing
## License

# Introduction

This notebook provides a practical implementation of two important aspects of NLP:

## Tokenization: 
   Tokenization is a fundamental step in NLP, where text is split into smaller units such as words or characters. Inspired by Andrej Karpathy's approach, this notebook demonstrates a simple yet effective way to tokenize text at the character level.

## Model Quantization: 
   Quantization is a technique used to reduce the memory footprint and increase the inference speed of machine learning models. In this notebook, we demonstrate how to convert a pre-trained DistilBERT model to int8 using the Hugging Face Optimum library and ONNX Runtime.

# Prerequisites

Before running the notebook, ensure you have the following installed:

### Python 3.6 or higher
### Jupyter Notebook or Google Colab
### Libraries:
#### transformers
#### optimum
#### onnx
#### onnxruntime

# Installation

## Using pip

To install the required Python packages, run:

```
pip install transformers optimum[onnxruntime] onnx onnxruntime
```

## Using Google Colab

You can also run the notebook in Google Colab without needing to install anything locally. Simply upload the notebook to Google Colab and execute the cells.

# Usage

## Running the Notebook

### 1. Clone the repository:

```
git clone https://github.com/hammadhaideer/Language-Model-Quantization.git
cd Language-Model-Quantization
```
### 2. Open the notebook:

```
jupyter notebook Language Model Quantization and Tokenization.ipynb
```
### 3. Run the cells in the notebook sequentially to perform tokenization and model quantization.

# Notebook Sections

## Section 1: Replicating the Andrej Karpathy Tokenizer Code

In this section, we replicate a simple tokenizer inspired by Andrej Karpathy. The tokenizer splits text into individual characters and maps them to unique indices. It also provides functionality to encode and decode text.

### Key Functions:
#### SimpleTokenizer: Class to handle tokenization.
#### encode(): Converts text to a list of indices.
#### decode(): Converts a list of indices back to text.

## Section 2: Convert a Language Model to int8 Using Optimum

This section demonstrates how to convert a pre-trained DistilBERT model to int8 using ONNX Runtime and the Hugging Face Optimum library. This process helps in optimizing the model for faster inference and lower memory usage.

### Key Steps:
#### Export the model to ONNX format.
#### Apply dynamic quantization to reduce the model size.
#### Verify the quantized model.

# Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

# License

This project is licensed under the MIT License. See the LICENSE file for details.
