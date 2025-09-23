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


def _simulate_short(row: pd.Series, entry_price: float, target_return: float) -> float:
    if target_return >= 0:
        return 0.0

    target_price = max(entry_price * (1.0 + target_return), 1e-6)
    stop_return = abs(target_return) / 10.0
    stop_price = max(entry_price * (1.0 + stop_return), 1e-6)

    for day in (1, 2, 3):
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

        if day == 3 and _valid_price(close_price):
            return 1.0 - float(close_price) / entry_price

    return 0.0


# Trading strategy based on quantile predictions and 4-day management rules
def trading_rule(test_df: pd.DataFrame, qh: float, ql: float) -> pd.DataFrame:
    df = test_df.copy()
    df["return"] = 0.0

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

        prev_low = row.get("BIDLO_-4d")
        prev_high = row.get("ASKHI_-4d")

        long_gate = (
            (entry_price - prev_low) / prev_low
            if prev_low is not None else None
        )

        short_gate = (
            (entry_price - prev_high) / prev_high
            if prev_high is not None else None
        )

        if (
            upper_pred is not None
            and not pd.isna(upper_pred)
            and upper_pred > 0
            and long_gate is not None
            and long_gate > 0
            and long_gate <= float(upper_pred)
        ):
            trade_return = _simulate_long(
                row, entry_price, prev_low, float(upper_pred))

        # Version A (strict): require the realized drop to reach 30% of the predicted lower bound magnitude.
        # cond_short_a = (
        #     lower_pred is not None
        #     and not pd.isna(lower_pred)
        #     and lower_pred < 0
        #     and short_gate is not None
        #     and short_gate < 0
        #     and abs(short_gate) >= 0.3 * abs(float(lower_pred))
        # )

        # Version B (lenient): trigger once the realized drop exceeds 30% of the predicted lower bound (still negative).
        # cond_short_b = (
        #     lower_pred is not None
        #     and not pd.isna(lower_pred)
        #     and lower_pred < 0
        #     and short_gate is not None
        #     and short_gate < 0
        #     and short_gate >= 0.3 * float(lower_pred)
        # )

        # if cond_short_b:
        #     trade_return = _simulate_short(row, entry_price, float(lower_pred))
        df.at[idx, "return"] = trade_return

    df["total_return"] = df["return"].cumsum()
    return df
