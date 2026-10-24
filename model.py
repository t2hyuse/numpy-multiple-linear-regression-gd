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

# Step 9 - mse_gradient (not yet solved)
# TODO: implement

# Step 10 - normal_equation (not yet solved)
# TODO: implement

# Step 11 - initialize_weights (not yet solved)
# TODO: implement

# Step 12 - gd_step (not yet solved)
# TODO: implement

# Step 13 - epoch_train_val_losses (not yet solved)
# TODO: implement

# Step 14 - update_early_stop_state (not yet solved)
# TODO: implement

# Step 15 - init_training_state (not yet solved)
# TODO: implement

# Step 16 - run_one_epoch (not yet solved)
# TODO: implement

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

