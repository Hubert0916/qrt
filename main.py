import numpy as np

def main():
    print("Hello from qrt!")

def reliability_score(y, lower, upper):
    inside = (y >= lower) & (y <= upper)
    return np.mean(inside)

def dr_metric(rs, alpha):
    return rs - (1 - alpha)

def pi_width(lower, upper):
    return upper - lower

def aw_metric(lower, upper):
    return np.mean(pi_width(lower, upper))

def apis_metric(y, lower, upper, alpha):
    w = pi_width(lower, upper)
    # 局部偏離量
    dev = np.zeros_like(y, dtype=float)
    below = y < lower
    above = y > upper
    dev[below] = lower[below] - y[below]
    dev[above] = y[above] - upper[above]

    penalty = (2.0 / alpha) * dev

    return np.mean(w + penalty)

def pinball_loss(y, pred, q):
    r = y - pred
    return np.mean(np.where(r > 0, q * r, (q - 1) * r))

if __name__ == "__main__":
    main()
