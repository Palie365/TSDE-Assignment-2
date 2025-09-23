import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=36)
data1_df = pd.read_csv('data_tsde_assignment_2_part_1.csv')
data2_df = pd.read_csv('data_tsde_assignment_2_part_2.csv')

data1 = data1_df['GDP_QGR'].values.flatten()

T1 = len(data1)


def part1_1():
    # ts plot
    plt.figure(figsize=(10, 4))
    plt.plot(data1, color='#4682b4', linewidth=1.5)
    plt.title("Dutch GDP Quarterly Growth Rate")
    plt.xlabel("Time")
    plt.ylabel("GDP Growth (%)")
    plt.grid(True, alpha=0.1)
    plt.show()

    # ACF plot
    plot_acf(data1, lags=12, alpha=0.05, zero=True, color='#cd5c5c')
    plt.title("SACF (lags 0-12)")
    plt.grid(True, alpha=0.1)
    plt.show()

    # ACF values
    acf_vals = acf(data1, nlags=12, fft=False)
    print("SACF values:")
    for lag, val in enumerate(acf_vals):
        print(f"Lag {lag}: {val:.4f}")


def part1_2():
    bic_values = []
    models = {}

    # AR(p) models with p = 1, 2, 3, 4
    for p in range(1, 5):
        model = sm.tsa.ARIMA(data1, order=(p, 0, 0), trend="c")
        results = model.fit()
        bic_values.append(results.bic)
        models[p] = results
        print(f"AR({p}) BIC: {results.bic:.2f}")

    # Select best p by minimum BIC
    best_p = np.argmin(bic_values) + 1
    best_model = models[best_p]

    print(f"\nSelected model: AR({best_p})")
    print(best_model.summary())

    return best_model, best_p


if __name__ == "__main__":
    part1_1()
    best_model, best_p = part1_2()
