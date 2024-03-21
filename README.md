# Defense Against Black Box Adversarial Attacks

This repository contains the implementation of the defense mechanism against black box adversarial attacks as described 
in the paper "Attacking Adversarial Attacks as A Defense". The implementation serves as a preprocessor for neural network models,
specifically designed to integrate with the Adversarial Robustness Toolbox (ART) and is validated for use with PyTorch models.

## Introduction

Adversarial attacks present a significant challenge to the reliability of machine learning models, 
especially in security-critical applications. The implemented defense mechanism enhances model 
robustness by preprocessing input data to mitigate the effects of black box adversarial attacks. 
This solution is tailored for PyTorch models, leveraging the ART library's preprocessing capabilities.

## Requirements

Python 3.x
PyTorch
Adversarial Robustness Toolbox (ART)
Ensure that you have the latest versions of PyTorch and ART installed in your environment to avoid compatibility issues.

## Installation

To use this defense mechanism in your project, first, ensure that ART is installed:
_pip install adversarial-robustness-toolbox_ 
Next, clone this repository to your local machine:
_git clone <repository-url>_ 

## Usage

After installing the required libraries and cloning the repository, 
you can integrate the defense preprocessor into your PyTorch model pipeline as follows:
- Import the preprocessor from the repository.
- Initialize the preprocessor with the necessary parameters.
- Attach the preprocessor to your model using ART's preprocessing functionality.
