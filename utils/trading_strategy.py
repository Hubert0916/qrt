import pandas as pd
import numpy as np


def _valid_price(value) -> bool:
    """
    Check if the price value is a valid positive number.

    Args:
        value: The price value to check.

    Returns:
        bool: True if valid and positive, False otherwise.
    """
    try:
        return pd.notna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _simulate_long(
    row: pd.Series,
    entry_price: float,
    previous_low: float,
    raw_prediction: float
) -> float:
    """
    Simulate a long trade execution over a 3-day holding period (Days 2-4).

    Args:
        row (pd.Series): Row containing price data.
        entry_price (float): The entry price (Open of Day 1).
        previous_low (float): Low price of the reference period (Day -4).
        raw_prediction (float): The model's predicted return.

    Returns:
        float: The percentage return of the trade.
    """
    if raw_prediction <= 0:
        return 0.0

    # Expansion Multiplier: Since QRF averages out predictions, it tends
    # to underestimate the magnitude of the move. We expand the target
    # slightly (e.g., 1.5x) to let profits run.
    target_return = raw_prediction * 1.5

    # Set target price relative to the previous low
    target_price = max(previous_low * (1.0 + target_return), 1e-6)

    # Standard Stop Loss (Tighten if needed)
    stop_price = max(entry_price * 0.9, 1e-6)

    for day in (2, 3, 4):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        # Check for Gap Openings
        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price >= target_price:
                return open_price / entry_price - 1.0
            if open_price <= stop_price:
                return open_price / entry_price - 1.0

        # Check Intraday Highs/Lows
        if _valid_price(high_price) and float(high_price) >= target_price:
            return target_price / entry_price - 1.0
        if _valid_price(low_price) and float(low_price) <= stop_price:
            return stop_price / entry_price - 1.0

        # Force Exit on Close of Day 4
        if day == 4 and _valid_price(close_price):
            return float(close_price) / entry_price - 1.0

    return 0.0


def _simulate_short(
    row: pd.Series,
    entry_price: float,
    previous_high: float,
    raw_prediction: float
) -> float:
    """
    Simulate a short trade execution over a 3-day holding period (Days 2-4).

    Args:
        row (pd.Series): Row containing price data.
        entry_price (float): The entry price (Open of Day 1).
        previous_high (float): High price of the reference period (Day -4).
        raw_prediction (float): The model's predicted return (negative value).

    Returns:
        float: The percentage return of the trade.
    """
    if raw_prediction >= 0:
        return 0.0

    # Expansion Multiplier for Shorts
    target_return = raw_prediction * 1.5

    # Set target price relative to the previous high
    target_price = max(previous_high * (1.0 + target_return), 1e-6)
    stop_price = max(entry_price * 1.10, 1e-6)

    for day in (2, 3, 4):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        # Check for Gap Openings
        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price <= target_price:
                return 1.0 - open_price / entry_price
            if open_price >= stop_price:
                return 1.0 - open_price / entry_price

        # Check Intraday Highs/Lows
        if _valid_price(low_price) and float(low_price) <= target_price:
            return 1.0 - target_price / entry_price
        if _valid_price(high_price) and float(high_price) >= stop_price:
            return 1.0 - stop_price / entry_price

        # Force Exit on Close of Day 4
        if day == 4 and _valid_price(close_price):
            return 1.0 - float(close_price) / entry_price

    return 0.0


def trading_rule(test_df: pd.DataFrame, qh: float, ql: float) -> pd.DataFrame:
    """
    Apply a trading strategy based on quantile predictions with risk management.

    This strategy incorporates dynamic gating and a skewness filter to leverage
    the risk-estimation capabilities of Quantile Regression Forests (QRF).

    Args:
        test_df (pd.DataFrame): DataFrame containing price data and predictions.
        qh (float): The upper quantile (e.g., 0.7 or 0.9).
        ql (float): The lower quantile (e.g., 0.3 or 0.1).

    Returns:
        pd.DataFrame: The input DataFrame with trade returns and cumulative
                      performance metrics appended.
    """
    df = test_df.copy()
    df["return"] = 0.0
    df["long_return"] = 0.0
    df["short_return"] = 0.0

    up_col = f"pred_q{qh}"
    low_col = f"pred_q{ql}"

    for idx, row in df.iterrows():
        entry_price = row.get("OPENPRC_1d")
        if not _valid_price(entry_price):
            continue
        entry_price = float(entry_price)

        upper_pred = row.get(up_col)  # Predicted Upside (e.g., +0.02)
        lower_pred = row.get(low_col)  # Predicted Downside (e.g., -0.01)

        # Skip if predictions are missing
        if pd.isna(upper_pred) or pd.isna(lower_pred):
            continue

        upper_pred = float(upper_pred)
        lower_pred = float(lower_pred)

        trade_return = 0.0
        long_return = 0.0
        short_return = 0.0

        prev_low = row.get("BIDLO_-4d")
        prev_high = row.get("ASKHI_-4d")
        past_return = row.get("past_return")

        # --- Dynamic Gating Logic ---
        long_gate_threshold = 0.0
        short_gate_threshold = 0.0

        # Calculate recent volatility/magnitude proxy
        volatility_proxy = 0.0
        if past_return is not None and pd.notna(past_return):
            volatility_proxy = abs(float(past_return))
        elif _valid_price(prev_low) and _valid_price(prev_high):
            volatility_proxy = (
                (float(prev_high) - float(prev_low)) / float(prev_low)
            )

        # Lower the barrier for QRF: Instead of requiring Prediction > Volatility,
        # we only require Prediction > 0.3 * Volatility. This helps
        # conservative QRF models enter trades.
        entry_sensitivity = 0.3
        long_gate_threshold = volatility_proxy * entry_sensitivity
        short_gate_threshold = volatility_proxy * entry_sensitivity
        risk_reward_ratio = 1.0

        # Skewness Filter: Check if upside potential outweighs downside risk.
        is_positive_skew = upper_pred > (abs(lower_pred) * risk_reward_ratio)
        is_negative_skew = abs(lower_pred) > (upper_pred * risk_reward_ratio)

        # --- Signal Generation ---

        # Long Signal Criteria:
        # 1. Predicted High is positive.
        # 2. Predicted High is significant enough (passes relaxed gate).
        # 3. Skew is positive (Upside > Downside risk).
        long_signal = (
            upper_pred > 0
            and upper_pred >= long_gate_threshold
            and is_positive_skew
        )

        # Short Signal Criteria:
        # 1. Predicted Low is negative.
        # 2. Predicted Low is significant enough.
        # 3. Skew is negative (Downside risk > Upside).
        short_signal = (
            lower_pred < 0
            and abs(lower_pred) >= short_gate_threshold
            and is_negative_skew
        )

        # --- Trade Execution ---

        if long_signal:
            if _valid_price(prev_low):
                long_return = _simulate_long(
                    row, entry_price, float(prev_low), upper_pred
                )

        if short_signal:
            if _valid_price(prev_high):
                short_return = _simulate_short(
                    row, entry_price, float(prev_high), lower_pred
                )

        # Conflict Resolution
        if long_signal and short_signal:
            # If both signals are valid (rare with skew filter),
            # pick the stronger one.
            if upper_pred > abs(lower_pred):
                trade_return = long_return
                short_return = 0.0
            else:
                trade_return = short_return
                long_return = 0.0
        elif long_signal:
            trade_return = long_return
            short_return = 0.0
        elif short_signal:
            trade_return = short_return
            long_return = 0.0

        # Store results
        df.at[idx, "return"] = trade_return
        df.at[idx, "long_return"] = long_return
        df.at[idx, "short_return"] = short_return

    # Calculate cumulative returns
    df["total_return"] = df["return"].cumsum()
    df["long_total_return"] = df["long_return"].cumsum()
    df["short_total_return"] = df["short_return"].cumsum()

    return df