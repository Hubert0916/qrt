import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Data from poster
quantile_pairs = ['ql=10, qh=90', 'ql=20, qh=80', 'ql=30, qh=70']
qrf_returns = [0.057, 0.061, 0.058]
qrt_returns = [0.012, 0.033, 0.022]

x = np.arange(len(quantile_pairs))
width = 0.32

plt.figure(figsize=(12, 6))
ax = plt.gca()

bars1 = ax.bar(x - width/2, qrf_returns, width, label='QRF')
bars2 = ax.bar(x + width/2, qrt_returns, width, label='QRT')

# Add value labels on bars (matching add_bar_labels style from benchmark.py)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.3f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
            f'{bar.get_height():.3f}%', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(quantile_pairs, rotation=30, ha='right')
ax.set_ylabel('Mean Annual Trading Performance')
ax.set_title('Annual Trading Performance: QRF vs QRT')
all_vals = qrf_returns + qrt_returns
ax.set_ylim(min(all_vals) * 0.95, max(all_vals) * 1.05)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend()

plt.tight_layout()
plt.savefig('output/annual_trading_performance.png', dpi=200)
plt.close()
print('Saved to output/annual_trading_performance.png')
