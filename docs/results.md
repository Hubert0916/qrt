# Results — Quantile Trees & Forests

---

## 1) Key Findings

- **QRF is the best probabilistic forecaster:** lowest average **pinball (↓)** across windows and **coverage** closer to the nominal 0.4; nearly insensitive to the split rule.
- **QRT/r² attains the highest P&L:** under our long-only rule, QRT with r² splits delivers the top average final cumulative return. **Trading performance ≠ better quantile accuracy** (different objective).
- **Robustness:** QRF behaves similarly across split rules; QRT is more sensitive to both the split rule and market window (regime spikes appear in some windows).

---

## 2) Main Results (figures + table)

### 2.1 Key Figures (Overview)
<p align="center">
  <img src="../output/benchmark_split/01_pinball_sum_bar.png" alt="Figure 1 — Pinball Loss" width="900"><br>
  <b>Figure 1.</b> <i>Pinball Loss (lower is better). QRF is lowest and robust to the split rule.</i>
</p>

<p align="center">
  <img src="../output/benchmark_split/02_coverage_bar.png" alt="Figure 2 — Coverage" width="900"><br>
  <b>Figure 2.</b> <i>Coverage between ql=0.3 and qh=0.7 (target ≈ 0.4). QRF is closer to nominal.</i>
</p>

<p align="center">
  <img src="../output/benchmark_split/03_cum_return_bar.png" alt="Figure 3 — Mean Final Cumulative Return" width="900"><br>
  <b>Figure 3.</b> <i>QRT/r² attains the highest average P&L under our long-only rule.</i>
</p>


### 2.2 Aggregated performance (means across windows)

**Metrics:** **Pinball (↓)**, **Coverage (→ 0.40)**, **Calibration Gap = |Coverage − 0.40| (↓)**, **Mean CumRet (↑)**.  
**Windows:** `nWindows = 15`.

| Model | Split | Pinball (↓) | Coverage (→0.40) | Calib. Gap (↓) | Mean CumRet (↑) | nWindows |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| QRF | loss | **0.1531** | 0.3533 | 0.0467 | 5.05 | 15 |
| QRF | mse  | **0.1531** | 0.3539 | **0.0461** | 5.04 | 15 |
| QRF | r²   | **0.1531** | 0.3539 | **0.0461** | 5.05 | 15 |
| QRT | loss | 0.1610 | 0.3000 | 0.1000 | 4.74 | 15 |
| QRT | mse  | 0.1610 | 0.3412 | 0.0588 | 4.78 | 15 |
| QRT | r²   | 0.1653 | 0.3097 | 0.0903 | **10.20** | 15 |

**Interpretation.**  
- **QRF:** pinball is minimal and nearly identical across split rules; coverage ≈ **0.354**, gap to nominal 0.4 ≈ **0.046** (slightly under-covered).  
- **QRT:** higher pinball and more criterion sensitivity; **r²** maximizes P&L (**10.20**) but departs further from nominal coverage—**misaligned with the quantile objective**.

---

## 3) Model-level comparison & robustness

Averaged over split rules, **QRF** has better probabilistic quality (lower pinball, smaller calibration gap, closer to nominal coverage), while **QRT** attains higher mean P&L.

### Model-Level Averages

Mean metrics collapsed across all split criteria for each model.

| Model | Pinball (↓) | Coverage (→0.40) | Calib. Gap (↓) | Mean CumRet (↑) | nWindows |
| :---  | ---: | ---: | ---: | ---: | ---: |
| **QRF** | **0.1531** | **0.3537** | **0.0463** | 5.05 | 15 |
| **QRT** | 0.1624 | 0.3170 | 0.0830 | **6.57** | 15 |

<p align="center">
  <img src="../output/advanced/pinball_sum_per_window.png"
       alt="Per-window pinball (ql+qh)" width="900"><br>
  <b>Figure 4.</b> <i>Per-window pinball (ql + qh).</i>
  QRF’s three curves almost overlap (robust to the split rule), while QRT shows higher variance and criterion sensitivity; a regime spike appears around W3.
</p>

---

## 4) Calibration & stability diagnostics

<p align="center">
  <img src="../output/advanced/04_signed_calibration_gap_per_window.png"
       alt="Signed calibration gap per window (coverage − (qh−ql))"
       width="900"><br>
  <b>Figure 5.</b> <i>Signed calibration gap per window (coverage − (qh−ql)).</i>
  QRF stays closer to 0 and within ±0.05 more often; QRT shows larger swings and frequent under-coverage (negative dips).
</p>

<p align="center">
  <img src="../output/advanced/coverage_vs_cum_return.png"
       alt="Coverage vs cumulative return per rolling window (scatter)"
       width="900"><br>
  <b>Figure 6.</b> <i>Coverage vs cumulative return (scatter).</i>
  High P&L can occur even when coverage falls below the nominal 0.4 — trading gains do not imply calibrated intervals.
</p>

<p align="center">
  <img src="../output/advanced/cum_return_per_window.png"
       alt="Final cumulative return per rolling window"
       width="900"><br>
  <b>Figure 7.</b> <i>Cumulative return per window.</i>
  Performance is window-dependent: QRT/r² shows a pronounced spike (e.g., W15), while QRF is steadier across windows.
</p>



**Pinball vs P&L (scatter)**  
Lower pinball does not necessarily lead to higher P&L — quantile accuracy and strategy performance are not monotonically aligned.
<p align="center">
  <img src="../output/advanced/pinball_vs_cum_return_no_outlier.png"
       alt="Pinball vs P&L (no outlier)" width="900"><br>
  <b>Figure 8.</b> <i>Pinball vs P&L (no outlier).</i>
</p>

<p align="center">
  <img src="../output/advanced/pinball_vs_cum_return.png"
       alt="Pinball vs P&L (full)" width="900"><br>
  <b>Figure 9.</b> <i>Pinball vs P&L (full view with outlier).</i>
</p>

---

## 5) Interpreting strategy performance (objective misalignment)

Our trading rule is **long-only**, with 4-day target/stop management, using **pred_q0.7** as the long trigger (with gating); the **short side is disabled**.  
Under this rule, **QRT/r²** may partition the samples into groups more favorable to this specific strategy, leading to higher P&L, while its split rule is **mean-oriented** and **not aligned** with quantile accuracy/calibration.  
Therefore, **higher P&L ≠ better quantile forecasts**. Interpret **probabilistic quality** (pinball/coverage) separately from **strategy performance**.

## 6) Conclusion

**TL;DR.** For probabilistic accuracy and calibrated intervals, **use QRF**; for this specific long-only rule, **QRT/r²** can yield higher average P&L, but that does **not** imply better quantile forecasts.

**What we learned**
- **Quantile accuracy.** QRF achieves the lowest **pinball** (≈ **0.153**) and the smallest **calibration gap** (≈ **0.046** to the nominal 0.4), and is largely **insensitive** to the split rule (loss/mse/r²).
- **Calibration over time.** The **signed calibration gap** (Fig. 5) shows QRF stays nearer to 0 and within ±0.05 more often; QRT exhibits larger negative dips (under-coverage) in several windows.
- **Trading performance.** Under our long-only gate/target/stop rule, **QRT/r²** delivers the highest mean final cumulative return (**10.20**), while QRF variants are steadier but lower (~**5.05**). This reflects **objective misalignment**: maximizing P&L for this rule is different from producing well-calibrated quantiles.

**Practical takeaway**
- If downstream decisions need **risk bounds / VaR-like intervals** or thresholds on **ql/qh**, prefer **QRF** (better-calibrated).
- If you only care about **this exact P&L rule**, QRT/r² can be competitive — but monitor **coverage** and **regime sensitivity**.
