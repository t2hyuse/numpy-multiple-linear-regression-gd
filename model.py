"""
NumPy Multiple Linear Regression GD

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - shuffle_xy
import numpy as np

def shuffle_xy(X, y, seed=42):
    """Randomly permute feature rows and targets together.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Feature matrix.
    y : np.ndarray, shape (n,)
        Target vector.
    seed : int, optional
        RNG seed for reproducibility (default 42).

    Returns
    -------
    X_shuffled : np.ndarray, shape (n, d)
    y_shuffled : np.ndarray, shape (n,)
    """

    rng = np.random.default_rng(seed)
    perm = rng.permutation(X.shape[0])
    return X[perm], y[perm]

# Step 2 - split_train_val_test
def split_train_val_test(X, y, train_frac=0.6, val_frac=0.2):
    # TODO: Slice already-shuffled data into contiguous train/val/test partitions...
    n = X.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return (X[0 : n_train, :],
    y[0 : n_train], 
    X[n_train : n_val + n_train, :],
    y[n_train : n_val + n_train],
    X[n_train + n_val:, :],
    y[n_train + n_val:])
    pass

# Step 3 - compute_feature_stats
def compute_feature_stats(X):
    # TODO: Compute per-feature mean and std; replace std of 0 with 1
    n, d = X.shape
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)

    for i in range(d):
        if std[i] == 0:
            std[i] = 1


    return mean, std
    pass

# Step 4 - standardize_features
def standardize_features(X, mean, std):
    # TODO: Apply z-score normalization using precomputed training mean and std.
    return (X - mean) / std
    pass

# Step 5 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to feature matrix X
    n, d = X.shape
    ret = np.ones((n, d+1))
    ret[:,0] = 1.

    ret[:, 1:] = X 
    return ret
    pass

# Step 6 - prepare_design_matrix
def prepare_design_matrix(X, mean, std):
    # TODO: Standardize features then add the bias column to form the design matrix.
    inter = standardize_features(X, mean, std)
    return add_bias_column(inter)
    pass

# Step 7 - predict_linear
def predict_linear(X, weights):
    """Compute linear predictions y_hat = X @ weights.

    Args:
        X: Design matrix of shape (n, d_in), often including a bias column.
        weights: Weight vector of shape (d_in,).

    Returns:
        Predicted targets of shape (n,).
    """
    # TODO: Return the predicted target vector from X and weights
    return X @ weights
    pass

# Step 8 - mse_loss
def mse_loss(y_true, y_pred):
    # TODO: Return the average of squared residuals as a scalar float.
    n = y_true.shape[0]
    return np.linalg.norm(y_true - y_pred)**2 / n 
    pass

# Step 9 - mse_gradient
def mse_gradient(X, y_true, y_pred):
    # TODO: Return the analytic MSE gradient w.r.t. weights: (2/n) X^T (y_pred - y_true)
    n = y_true.shape[0]
    return (2/n) * X.T @ (y_pred - y_true)
    pass

# Step 10 - normal_equation
def normal_equation(X, y):
    # TODO: Solve for the closed-form least-squares weights via the normal equation.
    return np.linalg.solve(X.T @ X, X.T @ y)
    pass

# Step 11 - initialize_weights
def initialize_weights(n_features, seed=None):
    """Return (n_features,) weights sampled from N(0, 0.01)."""
    rng = np.random.RandomState(seed)
    return rng.normal(0, 0.01, n_features)

# Step 12 - gd_step
def gd_step(X, y, weights, lr):
    """Run one full-batch gradient descent update on the weights.

    Args:
        X: Design matrix of shape (n, d_in).
        y: Target vector of shape (n,).
        weights: Current weight vector of shape (d_in,).
        lr: Learning rate (float).

    Returns:
        Updated weight vector of shape (d_in,).
    """
    # TODO: return the updated weight vector after one MSE gradient step

    pred = predict_linear(X, weights)
    grad = mse_gradient(X, y, pred)

    weights -= grad * lr
    return weights

    pass

# Step 13 - epoch_train_val_losses
import numpy as np

def epoch_train_val_losses(X_train, y_train, X_val, y_val, weights):
    """Evaluate MSE on train and validation sets for the current weights."""
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)

    train_loss = float(np.mean((X_train @ w - y_train) ** 2))
    val_loss = float(np.mean((X_val @ w - y_val) ** 2))
    return train_loss, val_loss

# Step 14 - update_early_stop_state
def update_early_stop_state(val_loss, best_val_loss, wait, weights, best_weights, patience):
    """Update early-stopping bookkeeping after one validation evaluation.

    Returns:
        (best_val_loss, wait, best_weights, stop)
    """
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = np.array(weights, copy=True)
        wait = 0
        stop = False
    else:
        wait += 1
        stop = wait >= patience
    return best_val_loss, wait, best_weights, stop

# Step 15 - init_training_state
def init_training_state(n_features, seed=None):
    weights = initialize_weights(n_features, seed)
    return {
        "weights": weights,
        "best_weights": np.array(weights, copy=True),
        "best_val_loss": np.inf,
        "wait": 0,
        "train_losses": [],
        "val_losses": [],
        "stopped": False,
    }

# Step 16 - run_one_epoch
def run_one_epoch(state, X_train, y_train, X_val, y_val, lr, patience):
    """Perform one GD step, log losses, and refresh early-stopping on state."""

    state["weights"] = gd_step(X_train, y_train, state["weights"], lr)


    train_loss, val_loss = epoch_train_val_losses(
        X_train, y_train, X_val, y_val, state["weights"]
    )
    state["train_losses"].append(train_loss)
    state["val_losses"].append(val_loss)

    best_val_loss, wait, best_weights, stop = update_early_stop_state(
        val_loss,
        state["best_val_loss"],
        state["wait"],
        state["weights"],
        state["best_weights"],
        patience,
    )
    state["best_val_loss"] = best_val_loss
    state["wait"] = wait
    state["best_weights"] = best_weights
    state["stopped"] = stop

    return state

# Step 17 - train_batch_gd (not yet solved)
# TODO: implement

# Step 18 - mean_absolute_error (not yet solved)
# TODO: implement

# Step 19 - root_mean_squared_error (not yet solved)
# TODO: implement

# Step 20 - r_squared (not yet solved)
# TODO: implement

# Step 21 - evaluate_regression (not yet solved)
# TODO: implement

# Step 22 - learning_curve_data (not yet solved)
# TODO: implement

# Step 23 - weights_l2_distance (not yet solved)
# TODO: implement

# Step 24 - create_lr_model (not yet solved)
# TODO: implement

# Step 25 - fit_lr_model (not yet solved)
# TODO: implement

# Step 26 - predict_lr_model (not yet solved)
# TODO: implement

# Step 27 - score_lr_model (not yet solved)
# TODO: implement

# Step 28 - compare_with_normal_equation (not yet solved)
# TODO: implement

