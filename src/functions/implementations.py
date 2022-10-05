# -*- coding: utf-8 -*-
import logging

import numpy as np
from .helpers import check_dimensions, calculate_mse, compute_err, chunks, calculate_gradient_logistic, calculate_loss_logistic, soft_treshold

_TEST_PARAMETERS_ = [
    {
        'base_model': 'least_squares_GD',
        'max_iters': 1000,
        'gamma': 0.1,
        'ridge_lambda' : 0.0,
        'tol': 1e-3,
    },
    {
        'base_model': 'least_squares_SGD',
        'max_iters': 1000,
        'gamma': 1e-3,
        'ridge_lambda' : 0.0,
        'tol': 1e-3,
    },
    {
        'base_model': 'least_squares',
        'max_iters': 0,
        'gamma': 0,
        'ridge_lambda' : 0,
        'tol': 0,
    },
    {
        'base_model': 'ridge_regression',
        'max_iters': 0,
        'gamma': 0,
        'ridge_lambda' : 1.0,
        'tol': 1e-3,
    },
    {
        'base_model': 'logistic_regression',
        'max_iters': 3000,
        'gamma': 1e-6,
        'ridge_lambda' : 0,
        'tol': 1e-3,
    },
    {
        'base_model': 'reg_logistic_regression',
        'max_iters': 3000,
        'gamma': 1e-6,
        'ridge_lambda' : 1.0,
        'tol': 1e-3,
    }
]

def least_squares_GD(y, tx, initial_w = None, max_iters = 1000, gamma = 1e-3, tol = 1e-3):
    """Solve the least squares equation by the method of gradient descent.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    initial_w : array_like, shape =  [n_samples], optional
        Initial weights. If inital_w is None it is initialized with zeros.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    gamma : float, optional, default 1e-3
        Step size.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked
        with respect to L2 norm of loss function gradient.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        Mean Squared Error(MSE) at last iteration.


    """
    # Initialization
    N, M = check_dimensions(y, tx)

    if initial_w is None:
        w = np.zeros(M)
    else:
        w = initial_w

    logging.debug("least_squares_GD: max_iters={0}, gamma={1}, tol={2}.".format(max_iters, gamma, tol))

    # Optimization loop
    for n_iter in range(max_iters):
        err = compute_err(y, tx, w)
        grad = -tx.T.dot(err) / N

        # Check convergence
        if (np.linalg.norm(grad) <= tol):
            loss = calculate_mse(err)
            logging.debug("Gradient descent method: converged in {0} iterations.\n".format(n_iter))
            return w, loss

        w -= gamma * grad

    loss = calculate_mse(err)
    logging.debug("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of gradient at last iteration is {0}.\n".format(np.linalg.norm(grad)))
    return w, loss


def least_squares_SGD(y, tx, initial_w = None, max_iters = 1000, gamma = 1e-3, tol = 1e-3):
    """Solve the least squares equation by the method of stochastic gradient descent.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    initial_w : array_like, shape =  [n_samples], optional
        Initial weights. If inital_w is None it is initialized with zeros.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    gamma : float, optional, default 1e-3
        Step size.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked
        with respect to L2 norm of loss function gradient.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        Mean Squared Error(MSE) at last iteration.


    """
    # Initialization
    N, M = check_dimensions(y, tx)

    if initial_w is None:
        w = np.zeros(M)
    else:
        w = initial_w

    batch_size = 1

    logging.debug("least_squares_SGD: max_iters={0}, gamma={1}, tol={2}.".format(max_iters, gamma, tol))

    # Optimization loop
    for n_iter in range(max_iters):
        for chunk in chunks(np.random.permutation(len(y)), batch_size):
            y_chunk, tx_chunk = y[chunk], tx[chunk]
            err = compute_err(y_chunk, tx_chunk, w)
            grad = -tx_chunk.T.dot(err) / len(y_chunk)

            # Check convergence
            if (np.linalg.norm(grad) <= tol):
                loss = calculate_mse(err)
                logging.debug("Gradient descent method: converged in {0}.\n".format(n_iter))
                return w, loss

            # Update weights
            w -= gamma * grad

    loss = calculate_mse(err)
    logging.debug("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of gradient at last iteration is {0}.\n".format(np.linalg.norm(grad)))
    return w, loss


def least_squares(y, tx):
    """Solve the least squares equation solving a linear system.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        Mean Squared Error(MSE).


    """
    _, _ = check_dimensions(y, tx)

    logging.debug("least_squares\n")

    # Compute weights
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    # Compute loss
    err = compute_err(y, tx, w)
    loss = calculate_mse(err)

    return w, loss


def ridge_regression(y, tx, lambda_ = 1.0):
    """Solve the ridge equation by solving a linear system.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    lambda_ : float, optional, default 1.0
        Regularization strength; must be a positive float.
        If lambda_ = 0.0 the ridge equation is equivalent to
        least squares equation.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        MSE + lambda_*||w||_2^2


    """
    N, M = check_dimensions(y, tx)

    logging.debug("ridge_regression: lambda = {0}.\n".format(lambda_))

    aI = lambda_*2*N * np.identity(M)
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    err = compute_err(y, tx, w)
    loss = calculate_mse(err) + lambda_ * np.linalg.norm(w)**2

    return w, loss


def logistic_regression(y, tx, initial_w = None, max_iters = 1000, gamma = 1e-3, tol = 1e-3):
    """Solve the logistic equation by the method of gradient descent.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    initial_w : array_like, shape =  [n_samples], optional
        Initial weights. If inital_w is None it is initialized with zeros.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    gamma : float, optional, default 1e-3
        Step size.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked
        with respect to L2 norm of loss function gradient.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        Logistic regression loss.


    """
    # Initialization
    N, M = check_dimensions(y, tx)

    if initial_w is None:
        w = np.zeros(M)
    else:
        w = initial_w

    logging.debug("Logistic regression: max_iters={0}, gamma={1}, tol={2}.".format(max_iters, gamma, tol))

    # Optimization loop
    for n_iter in range(max_iters):
        grad = calculate_gradient_logistic(y, tx, w)

        # Check convergence
        if (np.linalg.norm(grad) <= tol):
            loss = calculate_loss_logistic(y, tx, w)
            logging.debug("Gradient descent method: converged in {0} iterations.\n".format(n_iter))
            return w, loss

        # Update weights
        w -= gamma * grad

    loss = calculate_loss_logistic(y, tx, w)
    logging.debug("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of gradient at last iteration is {0}.\n".format(np.linalg.norm(grad)))
    return w, loss


def reg_logistic_regression(y, tx, lambda_ = 1.0, initial_w = None, max_iters = 1000, gamma = 1e-3, tol = 1e-3):
    """Solve the regularized logistic equation by the method of gradient descent.


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    lambda_ : float, optional, default 1.0
        Regularization strength; must be a positive float.
        If lambda_ = 0.0 the regularized equation is equivalent to
        logistic equation.

    initial_w : array_like, shape =  [n_samples], optional
        Initial weights. If inital_w is None it is initialized with zeros.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    gamma : float, optional, default 1e-3
        Step size.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked
        with respect to L2 norm of loss function gradient.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        Logistic regression loss + lambda_/2 * ||w||_2^2.


    """
    # Initialization
    N, M = check_dimensions(y, tx)

    if initial_w is None:
        w = np.zeros(M)
    else:
        w = initial_w

    logging.debug("Regularized logistic regression: lambda_={0}, max_iters={1}, gamma={2}, tol={3}".format(lambda_, max_iters, gamma, tol))

    # Optimization loop
    for n_iter in range(max_iters):
        grad = calculate_gradient_logistic(y, tx, w) + lambda_ * w

        # Check convergence
        if (np.linalg.norm(grad) <= tol):
            loss = calculate_loss_reg_logistic(y, tx, w) + lambda_/2.0 * np.linalg.norm(w)**2
            logging.debug("Gradient descent method: converged in {0} iterations".format(n_iter))
            return w, loss

        # Update weights
        w -= gamma * grad

    loss = calculate_loss_logistic(y, tx, w) + lambda_/2.0 * np.linalg.norm(w)**2
    logging.debug("Maximum number of iterations exceeded, stopping criteria not satisified. Norm of gradient at last iteration is {0}.\n".format(np.linalg.norm(grad)))
    return w, loss


def lasso_shooting(y, tx, lambda_ = 1.0, initial_w = None, max_iters = 1000, tol = 1e-3):
    """Solve the lasso equation using the shooting algorithm.

    Based on "Machine Learning: A Probabilistic Perspective, Kevin P. Murphy, The MIT Press (2012),
    page 441"


    Parameters
    ----------
    y : array-like, shape = [n_samples]
        Target values.

    tx : array-like, shape = [n_samples, n_features]
        Training data.

    lambda_ : float, optional, default 1.0
        Regularization strength; must be a positive float.
        If lambda_ = 0.0 the lasso equation is equivalent to
        least squares.

    initial_w : array_like, shape =  [n_samples], optional
        Initial weights. If inital_w is None it is initialized with ridge
        regression solution.

    max_iter : int, optional, default 1000
        Maximum number of iterations.

    tol : float, optional, default 1e-3
        Precision of the solution. Convergence is checked with respect to
        the difference in loss between two consecutive iterations.

    Returns
    -------
    w : array, shape = [n_features]
        Weights vector.
    loss : float
        MSE + lambda_*||w||_1.


    """
    # Initialization
    N, M = check_dimensions(y, tx)

    if initial_w is None:
        # Initialize weights with ridge regression solution
        w, _ = ridge_regression(y, tx, lambda_)
    else:
        w = initial_w

    # Pre-compute to save some multiplications
    tx_T_y = 2 * tx.T.dot(y)
    A = 2 * tx.T.dot(tx)

    # Optimization loop
    loss = 0.0
    for n_iter in range(max_iters):
        loss_old = loss
        w_old = w

        # Update weights, using a random choice
        # TODO choose the element with steepest gradient
        for j in np.random.permutation(M):

            a = A[j,j]
            if a != 0:
                c = tx_T_y[j] - np.sum(A[j,:].dot(w)) + A[j,j]*w[j];
                w[j] = soft_treshold(c/a, lambda_/a)
            else:
                w[j] = 0

        err = compute_err(y, tx, w)
        loss = calculate_mse(err) + lambda_ * np.linalg.norm(w, 1)

        # Check convergence
        converged = (np.abs(loss-loss_old) <= tol)
        if converged:
            logging.debug("Lasso shooting method: converged in {0} iterations".format(n_iter))
            return w, loss

    logging.debug("Maximum number of iterations exceeded, stopping criteria not satisified." )
    return w, loss
