import pandas as pd


def _valid_price(value) -> bool:
    try:
        return pd.notna(value) and float(value) > 0
    except (TypeError, ValueError):
        return False

# return the percentage


def _simulate_long(row: pd.Series, entry_price: float, previous_low: float, target_return: float) -> float:
    if target_return <= 0:
        return 0.0

    target_price = max(previous_low * (1.0 + target_return), 1e-6)
    stop_price = max(entry_price * 0.9, 1e-6)

    for day in (2, 3, 4):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price >= target_price:
                return open_price / entry_price - 1.0
            if open_price <= stop_price:
                return open_price / entry_price - 1.0

        if _valid_price(high_price) and float(high_price) >= target_price:
            return target_price / entry_price - 1.0
        if _valid_price(low_price) and float(low_price) <= stop_price:
            return stop_price / entry_price - 1.0

        if day == 4 and _valid_price(close_price):
            return float(close_price) / entry_price - 1.0

    return 0.0


def _simulate_short(row: pd.Series, entry_price: float, previous_high: float, target_return: float) -> float:
    if target_return >= 0:
        return 0.0

    target_price = max(previous_high * (1.0 + target_return), 1e-6)
    stop_price = max(entry_price * (1.10), 1e-6)

    for day in (2, 3, 4):
        open_price = row.get(f"OPENPRC_{day}d")
        high_price = row.get(f"ASKHI_{day}d")
        low_price = row.get(f"BIDLO_{day}d")
        close_price = row.get(f"PRC_{day}d")

        if day > 1 and _valid_price(open_price):
            open_price = float(open_price)
            if open_price <= target_price:
                return 1.0 - open_price / entry_price
            if open_price >= stop_price:
                return 1.0 - open_price / entry_price

        if _valid_price(low_price) and float(low_price) <= target_price:
            return 1.0 - target_price / entry_price
        if _valid_price(high_price) and float(high_price) >= stop_price:
            return 1.0 - stop_price / entry_price

        if day == 4 and _valid_price(close_price):
            return 1.0 - float(close_price) / entry_price

    return 0.0


# Trading strategy based on quantile predictions and 4-day management rules
def trading_rule(test_df: pd.DataFrame, qh: float, ql: float) -> pd.DataFrame:
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

        upper_pred = row.get(up_col)
        lower_pred = row.get(low_col)

        trade_return = 0.0
        long_return = 0.0
        short_return = 0.0

        prev_low = row.get("BIDLO_-4d")
        prev_high = row.get("ASKHI_-4d")
        past_return = row.get("past_return")

        long_gate = None
        short_gate = None
        if past_return is not None and pd.notna(past_return):
            past_return = float(past_return)
            if past_return > 0:
                long_gate = past_return
            else:
                short_gate = -past_return
        else:
            # Fallback to original logic if past_return is not available
            if _valid_price(prev_low):
                long_gate = (entry_price - float(prev_low)) / float(prev_low)
            if _valid_price(prev_high):
                short_gate = (float(prev_high) - entry_price) / \
                    float(prev_high)
        long_signal = (
            upper_pred is not None
            and not pd.isna(upper_pred)
            and upper_pred > 0
            and long_gate is not None
            and long_gate > 0
            and long_gate <= float(upper_pred)
        )
        short_signal = (
            lower_pred is not None
            and not pd.isna(lower_pred)
            and lower_pred < 0
            and short_gate is not None
            and short_gate > 0
            and short_gate <= abs(float(lower_pred))
        )

        if long_signal:
            if _valid_price(prev_low):
                long_return = _simulate_long(
                    row, entry_price, float(prev_low), float(upper_pred)
                )

        if short_signal:
            if _valid_price(prev_high):
                short_return = _simulate_short(
                    row, entry_price, float(prev_high), float(lower_pred)
                )

        if long_signal and short_signal:
            long_strength = float(upper_pred)
            short_strength = abs(float(lower_pred))
            if long_strength > short_strength:
                trade_return = long_return
                short_return = 0.0
            elif short_strength > long_strength:
                trade_return = short_return
                long_return = 0.0
            else:
                # Tie-breaker defaults to short to reflect more conservative stance.
                trade_return = short_return
                long_return = 0.0
        elif long_signal:
            trade_return = long_return
            short_return = 0.0
        elif short_signal:
            trade_return = short_return
            long_return = 0.0

        df.at[idx, "return"] = trade_return
        df.at[idx, "long_return"] = long_return
        df.at[idx, "short_return"] = short_return

    df["total_return"] = df["return"].cumsum()
    df["long_total_return"] = df["long_return"].cumsum()
    df["short_total_return"] = df["short_return"].cumsum()
    return df
