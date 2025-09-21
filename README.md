# Quantile-Based Prediction using Regression Trees and Forests

<p align="center">
  <img width="2400" height="1200" alt="01_pinball_sum_bar" src="https://github.com/user-attachments/assets/68af85d0-0162-4ab6-be7f-dc0bcc636024" />
</p>


<p align="center">
  <b>Figure 1.</b> <i>Quantile Regression Accuracy</i><br>
  <strong>Quantile regression forest consistently achieves the lowest pinball loss and effect of split criterion on QRF is minimal</strong>. 
  This shows the advantage instability and generalization over single-tree models and results for <code>loss</code>, <code>mse</code>, and <code>r²</code> are nearly identical, highlighting the robustness of the ensemble approach.
</p>

<img width="2400" height="1200" alt="03_cum_return_bar" src="https://github.com/user-attachments/assets/1437151e-d342-49fb-a071-dde318f84ac5" />


<p align="center">
  <b>Figure 2.</b> <i>Cumulative return</i><br>
  TBD
</p>

## Motivation

Traditional regression models predict the **conditional mean**, but in many applications, the mean alone is insufficient to describe the true distribution. **Quantile Regression Models** instead predict conditional quantiles(e.g., 10th percentile, 90th percentile), allowing us to capture both the central tendency and tail behavior.

### Why Quantile Regression?

- **Asymmetric Risk Views**  
  In finance or risk management, tail events (extreme losses/gains) are crucial. Quantile regression can estimate metrics like Value-at-Risk or the probability of rare jumps/drops.
- **Prediction Intervals & Coverage**  
  Rather than a single point estimate, we care about intervals that cover the true outcome with a certain probability. For example, the model predicts that next-day return will lie in **[-1%, +2%]**, with **80% coverage**.
- **Decision Making Based on Upper/Lower Bounds**  
  For example, in trading strategies:  
  - Go long when the predicted upper quantile > 0  
  - Go short when the predicted lower quantile < 0  

This makes decisions not only based on the mean but also accounting for risk and opportunity.

Also this research compares:
- QRT — a single Quantile Regression Tree (fast, interpretable);
- QRF — an ensemble of quantile trees (Quantile Regression Forest; usually more stable).

And, critically, it compares three split criteria used to grow trees:
- Loss → minimize pinball (quantile) loss,
- MSE → minimize within-node SSE (classic CART),
- $R^2$ → maximize $R^2$ (SSE reduction ratio).

## Dataset
TBD

## Tree Optimization
- **Keep data as NumPy arrays + pass index views**
  - Store `X`(feature matrix), `y`(target matrix) once; each node carries only an `indices` view (zero-copy).

- **Single-pass threshold search per feature**
  - Sort samples **once** by the candidate feature, then scan all split points on this
    sorted array.
  - Turns “rebuild mask for every threshold” into “sort once + scan”.

- **Vectorized MSE using prefix sums**
  - Precompute cumulative count/sum/sum-of-squares; each split’s SSE (left+right) is computed in **O(1)** with closed-form formulas.

- **Cap candidate thresholds (K) with optional random subsampling**
  - If a feature has many unique values, evaluate at most **K** cut points.
  - Linear, predictable runtime with minimal accuracy loss (especially in forests).

- **Optional random feature subspace (√d per node)**
  - Evaluate splits on a subset of features (Random-Forest style).
  - Faster training + more diverse trees → often better generalization.
  

## Split Criteria

### 1. Pinball Loss

For residual $$u = y - \hat y$$:

$$
\rho_q(u) =
\begin{cases}
q \cdot u, & u \ge 0 \\
(1-q)(-u), & u < 0
\end{cases}
$$

We choose splits that **minimize** total pinball loss in left/right children.

### 2. Mean Squared Error

The best constant prediction (under squared error) is the mean $$\bar y$$.  
For candidate split $$S$$:

$$
\text{SSE}(S) = \sum_{i \in L}(y_i - \bar y_L)^2 + \sum_{i \in R}(y_i - \bar y_R)^2
$$

We choose the split that **minimizes** $$\text{SSE}(S)$$.

### 3. Coefficient of Determination $$R^2$$ 

$$
R^2(S) = 1 - \frac{\text{SSE}(S)}{\text{SST}}, \quad
\text{SST} = \sum_i (y_i - \bar y)^2
$$

We choose splits that **maximize** $$R^2(S)$$.


## Experiment Design
### Hyperparameters

All experiments use **default settings** unless otherwise specified.

#### Quantile Regression Tree (QRT)

| Parameter          | Value |
|-------------------|-------|
| `max_depth`       | 10    |
| `min_samples_leaf`| 5     |
| `random_state`    | 42    |

#### Quantile Regression Forest (QRF)

| Parameter          | Value |
|-------------------|-------|
| `n_estimators`    | 100   |
| `max_depth`       | 10    |
| `min_samples_leaf`| 5     |
| `random_state`    | 42    |

#### Rolling Window Setup

| Parameter        | Value |
|-----------------|-------|
| `train_period`  | 5 (years) |
| `test_period`   | 1 (year) |
| `ql`            | 0.3 |
| `qh`            | 0.7 |

### Trading Stategy

### Rolling-Window Setup

- **Training window**: $$N_\text{train}$$ years (default 5)  
- **Testing window**: $$N_\text{test}$$ years (default 1)  
- Slide the window forward and repeat.

Example: Train 2004–2008 → Test 2009, then Train 2005–209 → Test 2010, etc.

### Models & Criteria

- **QRT** (Quantile Regression Tree)
- **QRF** (Quantile Regression Forest — uses Bagging/Bootstrap Aggregation)

with `split_criterion ∈ {loss, mse, r2}`

Quantiles evaluated: $$q_l = 0.1, q_h = 0.9$$

### Metrics

- **Pinball loss**:  
  $$\rho_{q_l} + \rho_{q_h}$$ (lower is better)
- **Coverage**:  
  $$\frac{\left|\{y \mid \hat y_{ql} \le y \le \hat y_{qh}\}\right|}{\left|\{y\}\right|}$$, 
  Ideally close to $$q_h - q_l$$ if intervals are calibrated.
- **Cumulative return**:  
  Backtest a long/short strategy based on predicted quantile bounds.


##  Results

TBD


## Conclusion

TBD


## Future Work

...Model Tree...


