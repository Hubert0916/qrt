# Quantile-Based Prediction using Regression Trees and Forests

<p align="center">
  <img width="2400" height="1200" alt="01_pinball_sum_bar" src="https://github.com/user-attachments/assets/68af85d0-0162-4ab6-be7f-dc0bcc636024" />
</p>
<p align="center">
  <b>Figure 1.</b> <i>Quantile Regression Accuracy</i><br>
  <strong>Quantile regression forest consistently achieves the lowest pinball loss and effect of split criterion on QRF is minimal</strong>. 
  This shows the advantage instability and generalization over single-tree models and results for <code>loss</code>, <code>mse</code>, and <code>r²</code> are nearly identical, highlighting the robustness of      the ensemble approach.
</p>

<p align="center"> 
  <img width="2400" height="1200" alt="02_coverage_bar" src="https://github.com/user-attachments/assets/8c57dc1f-642c-4230-8776-714d4ea0f57b" />
</p>
<p align="center">
  <b>Figure 2.</b> <i>Interval Coverage between quantile=0.3 and 0.7</i><br>
  QRF shows consistently higher coverage (~35%) compared to QRT (30–34%), indicating better-calibrated prediction intervals. Split criterion has little impact on QRF performance.
</p>

<p align="center"> 
  <img width="2400" height="1200" alt="03_cum_return_bar" src="https://github.com/user-attachments/assets/1437151e-d342-49fb-a071-dde318f84ac5" />
</p>
<p align="center">
  <b>Figure 3.</b> <i>Mean Cumulative Return across Rolling Windows</i><br> 
  QRT with <code>r²</code>-based splitting achieved the highest trading performance, nearly doubling the cumulative return compared to all other configurations. QRF variants showed stable but lower returns.
</p>

<p align="center"> 
  <img width="2200" height="1320" alt="qrt_size_sweep" src="https://github.com/user-attachments/assets/0a91815e-bec0-4f5b-b6c2-984845b960ef" />
</p>
<p align="center">
  <b>Figure 4.</b> <i>Execution time vs. training size for old and optimized QRT implementations.</i><br>
  The optimized version consistently achieves <strong>20×–100× faster training</strong> (geometric mean speedup ≈<strong>60×</strong>) while maintaining similar pinball loss.
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


## Tree Optimization

### 1. Zero-Copy Data Handling
-  **Method:** Data is never copied. The full feature matrix `X` and target vector `y` are stored only once in memory. When a tree node splits, it passes a lightweight **index array** to its children, not a subset of the data. This array tells the child node which rows of the original data it is responsible for.
-  **Benefit:** This significantly reduces memory consumption and avoids the time overhead of data copying, allowing the model to handle much larger datasets.

### 2. Efficient Single-Pass Threshold Search

-  **Method:** To find the best split for a feature, a naive approach would be to re-scan the data for every possible threshold. Our implementation sorts the data **once** by the feature's values, while also reordering the target vector `y` in the same way. Afterwards, all candidate split points can be evaluated by scanning this single sorted array.

-   **Benefit:** This avoids massive amounts of redundant computation, transforming an expensive operation (re-scan for every threshold) into an efficient one (sort once, then scan once).

### 3. Vectorized MSE Calculation with Prefix Sums

-  **Method:** When using Mean Squared Error (MSE) as the splitting criterion, we employ a technique called **Prefix Sums**. We pre-compute the cumulative sum of sample counts, target values, and squared target values. With these pre-computed arrays, the Sum of Squared Errors (SSE) for any split can be calculated in **O(1) time** using a simple formula.

-  **Benefit:** This is an algorithmic leap that makes the process of scoring split points nearly instantaneous, resulting in an order-of-magnitude speedup.

### 4. Accelerated Quantile Calculation with `np.partition`

-  **Method:** When calculating Quantile Loss or $R^2$, we need to find a specific quantile of the data. A full sort, as used by `np.quantile`, takes `O(nlogn)` time. We use `np.partition` instead, which is based on the [Quickselect](https://en.wikipedia.org/wiki/Quickselect) algorithm. It only needs to find the k-th smallest element and place it in its correct sorted position, without sorting the rest of the array. Its average time complexity is just `O(n)`.

-  **Benefit:** `O(n)` is significantly faster than `O(nlogn)`. For large datasets, this change saves a substantial amount of time during quantile computation.

  

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
### Dataset

Our experiments are based on a **financial ESG dataset**, augmented with **text-derived features** and historical price information.

- **Time Range:** 2003–2023  
- **Samples:** ~97000 observations  
- **Features:** 500+ ESG-related TF-IDF features extracted from company filings  
- **Target:** Daily stock return (`報酬率`)

### Feature Composition

- **TF-IDF Features:**  
  Each sample contains a vector of word importance scores capturing ESG disclosure intensity.

- **Price History:**  
  Includes open, high, low, and close prices with multiple day lags/leads (e.g., `OPENPRC_1d`, `BIDLO_-4d`).

  
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

#### Signal Construction
- Each rolling test window is enriched with two columns: `pred_q0.3` and `pred_q0.7`, produced by the chosen quantile model.
- The entry candle is the row where `OPENPRC_1d` is observed; rows with invalid or missing prices are ignored.
- For a long candidate we compute a gating ratio `(OPENPRC_1d - BIDLO_-4d) / BIDLO_-4d`. The trade is eligible only when:
  - the upper quantile forecast is positive,
  - the previous four-day low exists and the ratio is positive, and
  - the ratio does not exceed the forecasted upper quantile. This keeps the breakout size consistent with what the model expects.
- No other filters (volume, trend, etc.) are currently applied; every row that survives the checks becomes a simulated long trade.

#### Long Trade Management
- The entry is executed at `OPENPRC_1d`.
- The profit target is derived from the previous low: `target_price = BIDLO_-4d * (1 + pred_q0.7)`. This ties the expected upside to how far the model believes price can rebound from the recent trough.
- A fixed 10% stop is placed immediately via `stop_price = OPENPRC_1d * 0.9`.
- The exit logic scans the next three sessions (labelled `OPENPRC_2d` through `OPENPRC_4d`, plus their corresponding high/low/close columns):
  - If the gap-open on day 2–4 hits either the target or the stop, the position is closed at that open.
  - Otherwise, intraday highs and lows are checked; touching the target or stop produces the respective return.
  - Any position still open at the end of day 4 is liquidated at `PRC_4d`.
- Realized percentage P&L is logged in the `return` column, and the cumulative series is tracked via `total_return`.

#### Short Side (Inactive)
Although a `_simulate_short` function has been implemented in the code, it is currently disabled.  
The reason is that our training set and feature design are based primarily on **long-side scenarios**,  
and there is insufficient short-side data to support meaningful backtesting.  

Therefore, all reported results and metrics are **long-only**.  

#### Reporting
- The trading function returns the per-row ledger with realized returns and `total_return` used in the benchmark metrics.
- These outputs feed into `metrics.csv`, where cumulative performance is compared across model/split-criterion combinations.


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


