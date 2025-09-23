### Rolling Window Metrics by Model/Split

Aggregated averages across all rolling windows for each model/split criterion combination.

| Model | Split | Pinball (↓) | Coverage (→0.40) | Calib. Gap (↓) | Mean CumRet (↑) | nWindows |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| QRF | loss | **0.1531** | 0.3533 | 0.0467 | 5.05 | 15 |
| QRF | mse | **0.1531** | 0.3539 | **0.0461** | 5.04 | 15 |
| QRF | r2 | **0.1531** | 0.3539 | **0.0461** | 5.05 | 15 |
| QRT | loss | 0.1610 | 0.3000 | 0.1000 | 4.74 | 15 |
| QRT | mse | 0.1610 | 0.3412 | 0.0588 | 4.78 | 15 |
| QRT | r2 | 0.1653 | 0.3097 | 0.0903 | **10.20** | 15 |


### Model-Level Averages

Mean metrics collapsed across all split criteria for each model.

| Model | Pinball (↓) | Coverage (→0.40) | Calib. Gap (↓) | Mean CumRet (↑) | nWindows |
| :--- | ---: | ---: | ---: | ---: | ---: |
| QRF | **0.1531** | 0.3537 | **0.0463** | 5.05 | 15 |
| QRT | 0.1624 | 0.3170 | 0.0830 | **6.57** | 15 |
