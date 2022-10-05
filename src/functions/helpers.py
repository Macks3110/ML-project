# -*- coding: utf-8 -*-
import numpy as np


def predict_labels(tx, w, logistic = False):
    """Applies linear model and transforms predictions into class labels"""
    y_pred = tx.dot(w)

    if logistic:
        y_pred = sigmoid(y_pred)
        true_idx = y_pred > 0.5
    else:
        true_idx = y_pred > 0

    y_pred[~true_idx] = -1
    y_pred[true_idx] = 1

    return y_pred
def check_dimensions(y, tx):
    """Check the dimensions of inputs"""

    N = np.shape(y)[0]
    Nx, M = np.shape(tx)

    if N == Nx:
        return N, M
    else:
        raise ValueError("Dimensions mismatch, please check your inputs.")


def chunks(l, n):
    """Yield successive n-sized chunks from l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def compute_err(y, tx, w):
    """Compute error vector."""
    return y - tx.dot(w)


def calculate_mse(e):
    """Calculate the Mean Squared Error(MSE) for vector e."""
    return np.mean(e**2) / 2.0


# Helpers for logistic
def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss_logistic(y, tx, w):
    """Compute logistic loss function."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)


def calculate_gradient_logistic(y, tx, w):
    """Compute the gradient of logistic loss function."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

# Helpers for LASSO shooting
def soft_treshold(a, delta):
    """Soft treshold operator"""
    a = np.sign(a) * np.maximum(np.abs(a) - delta, 0)
    return a
