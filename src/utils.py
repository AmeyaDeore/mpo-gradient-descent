import tensorflow as tf
import risk_measures as rm


def round_sigmoid(decimals=0):
    """
    Creates a custom gradient sigmoid function that rounds the output to a specified number of decimals.

    Args:
        decimals (int): Number of decimal places to round the output.

    Returns:
        function: A custom gradient sigmoid function.
    """

    @tf.custom_gradient
    def my_rounded_sigmoid(x):
        z = tf.nn.sigmoid(x)
        scale = 10**decimals
        output = tf.math.round(z * scale) / scale

        def backward(dy):
            custom_grad = z * (1 - z)
            return dy * custom_grad

        return output, backward

    return my_rounded_sigmoid


def exceeding_threshold(x, alpha):
    """
    Computes the ReLU of the difference between x and alpha.

    Args:
        x (tf.Tensor): Input tensor.
        alpha (float): Threshold value.

    Returns:
        tf.Tensor: Result of ReLU(x - alpha).
    """
    return tf.nn.relu(x - alpha)


def under_threshold(x, alpha):
    """
    Computes the ReLU of the difference between alpha and x.

    Args:
        x (tf.Tensor): Input tensor.
        alpha (float): Threshold value.

    Returns:
        tf.Tensor: Result of ReLU(alpha - x).
    """
    return tf.nn.relu(alpha - x)


def mask_greater_than(x, threshold, epsilon=1e-16):
    """
    Masks values greater than a threshold using a rounded sigmoid function.

    Args:
        x (tf.Tensor): Input tensor.
        threshold (float): Threshold value.
        epsilon (float, optional): Small value to avoid numerical issues (default is 1e-16).

    Returns:
        tf.Tensor: Masked tensor.
    """
    return round_sigmoid()(x - threshold - epsilon)


def mask_lower_than(x, threshold, epsilon=1e-16):
    """
    Masks values lower than a threshold using a rounded sigmoid function.

    Args:
        x (tf.Tensor): Input tensor.
        threshold (float): Threshold value.
        epsilon (float, optional): Small value to avoid numerical issues (default is 1e-16).

    Returns:
        tf.Tensor: Masked tensor.
    """
    return round_sigmoid()(threshold + epsilon - x)


def ConstraintUCITS_1(w):
    """
    Computes the sum of weights exceeding a threshold of 0.1.

    Args:
        w (tf.Tensor): Weight tensor.

    Returns:
        tf.Tensor: Sum of weights exceeding the threshold.
    """
    return tf.math.reduce_sum(exceeding_threshold(w, 0.1))


def ConstraintUCITS_2(w):
    """
    Computes a constraint based on weights lower than 0.05 and exceeding a sum threshold.

    Args:
        w (tf.Tensor): Weight tensor.

    Returns:
        tf.Tensor: Constraint value.
    """
    mask = mask_lower_than(w, 0.05)
    return exceeding_threshold(tf.math.reduce_sum(w * mask) - 0.4)


def ConstraintTrackingError(x, y, TE_max):
    """
    Computes the tracking error constraint.

    Args:
        x (tf.Tensor): Portfolio returns.
        y (tf.Tensor): Benchmark returns.
        TE_max (float): Maximum tracking error allowed.

    Returns:
        tf.Tensor: Constraint value.
    """
    return exceeding_threshold(rm.RiskMeasures.TrackingError(x, y), TE_max)


def ConstraintMinWeights(w, min_value):
    """
    Computes a constraint based on minimum weights.

    Args:
        w (tf.Tensor): Weight tensor.
        min_value (float): Minimum weight value.

    Returns:
        tf.Tensor: Constraint value.
    """
    mask = mask_greater_than(w, min_value)
    return tf.math.reduce_sum(w * mask)


def ConstraintRange(w, low, high):
    """
    Computes a constraint based on weight range.

    Args:
        w (tf.Tensor): Weight tensor.
        low (float): Lower bound.
        high (float): Upper bound.

    Returns:
        tf.Tensor: Constraint value.
    """
    mask = mask_lower_than(w, 0.0)
    lower = low - tf.math.reduce_sum(mask)
    higher = high - tf.math.reduce_sum(mask)
    return exceeding_threshold(lower * higher)


def ConstraintSubsets(w, M, m):
    """
    Computes a constraint based on subsets.

    Args:
        w (tf.Tensor): Weight tensor.
        M (tf.Tensor): Subset matrix.
        m (tf.Tensor): Subset vector.

    Returns:
        tf.Tensor: Constraint value.
    """
    return tf.math.reduce_sum(tf.abs((m - w @ M)))


def softmax(w):
    """
    Computes the softmax of a tensor along the specified axis.

    Args:
        w (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Softmax of the input tensor.
    """
    return tf.nn.softmax(w, axis=0)


def sparsemax(v, s=1, axis=0):
    """
    Projects a tensor onto the simplex defined by sum(w) = s and w >= 0 along the specified axis.

    Args:
        v (tf.Tensor): Tensor to project.
        s (int, optional): Desired sum of the projected components (default is 1).
        axis (int, optional): Axis along which the projection is performed (default is 0).

    Returns:
        tf.Tensor: Projected tensor.
    """
    # Sort v in descending order along the specified axis
    u = tf.sort(v, axis=axis, direction="DESCENDING")

    # Compute the cumulative sum along the specified axis
    cssv = tf.cumsum(u, axis=axis)

    # Create a tensor of indices
    rho = tf.range(1, tf.shape(v)[axis] + 1, dtype=v.dtype)

    # Reshape rho to match the dimensions of v
    rho_shape = [1] * len(v.shape)
    rho_shape[axis] = -1
    rho = tf.reshape(rho, rho_shape)

    # Compute the condition for each index
    condition = u + (s - cssv) / rho > 0

    # Get the indices that satisfy the condition
    rho_max = tf.reduce_sum(tf.cast(condition, tf.int32), axis=axis, keepdims=True)

    # Compute theta
    u_safe = tf.where(condition, u, tf.zeros_like(u))
    theta = (tf.reduce_sum(u_safe, axis=axis, keepdims=True) - s) / tf.cast(rho_max, v.dtype)

    # Project
    w = tf.maximum(v - theta, 0)
    return w
