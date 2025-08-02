"""
Performance evaluation metrics for quantile regression models.

This module contains functions to evaluate the quality of prediction intervals
and quantile predictions using various metrics.
"""

import numpy as np


def reliability_score(y, lower, upper):
    """
    Calculate the reliability score (coverage probability) of prediction intervals.

    Parameters
    ----------
    y : array-like
        True target values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Proportion of true values that fall within the prediction intervals.
    """
    inside = (y >= lower) & (y <= upper)
    return np.mean(inside)


def dr_metric(rs, alpha):
    """
    Calculate the deviation rate (Dr) metric.

    Parameters
    ----------
    rs : float
        Reliability score (coverage probability).
    alpha : float
        Nominal miscoverage rate (1 - confidence level).

    Returns
    -------
    float
        Difference between observed and expected coverage rates.
    """
    return rs - (1 - alpha)


def pi_width(lower, upper):
    """
    Calculate the width of prediction intervals.

    Parameters
    ----------
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    array-like
        Width of each prediction interval.
    """
    return upper - lower


def aw_metric(lower, upper):
    """
    Calculate the Average Width (AW) metric of prediction intervals.

    Parameters
    ----------
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Average width of prediction intervals.
    """
    return np.mean(pi_width(lower, upper))


def apis_metric(y, lower, upper, alpha):
    """
    Calculate the Average Prediction Interval Score (APIS) metric.

    This metric combines interval width with penalties for non-coverage.

    Parameters
    ----------
    y : array-like
        True target values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.
    alpha : float
        Nominal miscoverage rate (1 - confidence level).

    Returns
    -------
    float
        Average prediction interval score.
    """
    w = pi_width(lower, upper)

    # Calculate local deviation amounts
    dev = np.zeros_like(y, dtype=float)
    below = y < lower
    above = y > upper
    dev[below] = lower[below] - y[below]
    dev[above] = y[above] - upper[above]

    penalty = (2.0 / alpha) * dev

    return np.mean(w + penalty)


def pinball_loss(y, pred, q):
    """
    Calculate the pinball loss (quantile loss) function.

    Parameters
    ----------
    y : array-like
        True target values.
    pred : array-like
        Predicted quantile values.
    q : float
        Quantile level (between 0 and 1).

    Returns
    -------
    float
        Average pinball loss.
    """
    r = y - pred
    return np.mean(np.where(r > 0, q * r, (q - 1) * r))