import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=36)
data1_df = pd.read_csv('data_tsde_assignment_2_part_1.csv')
data2_df = pd.read_csv('data_tsde_assignment_2_part_2.csv')

data1 = data1_df['GDP_QGR'].values.flatten()

T1 = len(data1)


def part1_1():
    print("--- PART 1.1: Data Plot and ACF ---")
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
    print("\n--- PART 1.2: AR(p) Model Selection ---")
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


def part1_3_and_4(best_model, best_p, h=8):
    print("\n--- PART 1.3 & 1.4: Forecasting and Residual Analysis ---")
    const_hat = best_model.params[0]
    phi_hat = best_model.params[1]

    # forecasts
    forecasts = []
    X_T = data1[T1 - 1]
    for _ in range(h):
        next_val = const_hat + phi_hat * X_T
        forecasts.append(next_val)
        X_T = next_val
    forecasts = np.array(forecasts)

    # variance and standard error for CI
    var_hat = np.var(best_model.resid, ddof=best_p + 1)
    psi_sq_sum = np.cumsum(phi_hat ** (2 * np.arange(h)))
    se = np.sqrt(var_hat * psi_sq_sum)
    ci_lower = forecasts - 1.96 * se
    ci_upper = forecasts + 1.96 * se

    forecast_index = np.arange(T1, T1 + h)

    # plot with CI
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(T1), data1, color='#4682b4', linewidth=1.5, label="Observed")
    forecast_x = np.concatenate([[T1 - 1], forecast_index])
    forecast_y = np.concatenate([[data1[-1]], forecasts])
    plt.plot(forecast_x, forecast_y, color='#cd5c5c', linewidth=1.5, label="Forecast")
    plt.fill_between(forecast_index, ci_lower, ci_upper,
                     color='#cd5c5c', alpha=0.35, label="95% CI")
    plt.title(f"Forecasts with 95% CI (AR({best_p}), {h} Quarters Ahead)")
    plt.xlabel("Time")
    plt.ylabel("GDP Growth (%)")
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.show()

    print("Forecasts for next 8 quarters (with 95% CI):")
    for i, (f, lo, hi) in enumerate(zip(forecasts, ci_lower, ci_upper), start=1):
        print(f"h={i}: {f:.4f}  (95% CI: {lo:.4f}, {hi:.4f})")

    residuals = best_model.resid
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    print(f"\nJarque–Bera test: stat={jb_stat:.2f}, p-value={jb_p:.3f}")
    if jb_p < 0.05:
        print("Residuals deviate from normality (at 5% level).")
    else:
        print("Residuals consistent with normality (at 5% level).")

    k = int(np.sqrt(T1))
    bg_stat, bg_pval, _, _ = acorr_breusch_godfrey(best_model, nlags=k)
    print(f"Breusch–Godfrey test (lag={k}): stat={bg_stat:.2f}, p-value={bg_pval:.3f}")
    if bg_pval < 0.05:
        print("Residuals show autocorrelation (at 5% level).")
    else:
        print("Residuals are consistent with white noise (at 5% level).")


def dataloader2():
    gdp_df = data2_df[['GDP_QGR']]
    un_series = data2_df['UN_RATE']
    return gdp_df, un_series


def part2_1(gdp_df, un_series):
    print("\n--- PART 2.1: Data Plot and Model Estimation ---")
    plt.figure(figsize=(10, 4))
    plt.plot(gdp_df, color='tab:blue', linewidth=1.5, label='GDP Growth Rate')
    plt.plot(un_series, color='tab:orange', linewidth=1.5, label='Unemployment Rate')
    plt.title("Dutch GDP Quarterly Growth Rate and Unemployment Rate")
    plt.xlabel("Time")
    plt.ylabel("Rate (%)")
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.show()

    print("\n--- AR Model Selection for GDP Growth ---")
    aic_values_ar = []
    models_ar = {}
    for p in range(1, 5):
        model = sm.tsa.ARIMA(gdp_df, order=(p, 0, 0), trend="c")
        results = model.fit()
        aic_values_ar.append(results.aic)
        models_ar[p] = results
        print(f"AR({p}) AIC: {results.aic:.2f}")

    best_p_ar = np.argmin(aic_values_ar) + 1
    best_model_ar = models_ar[best_p_ar]
    print(f"\nSelected AR model for GDP: AR({best_p_ar})")
    print(best_model_ar.summary())

    print("\n--- ADL Model Selection for Unemployment ---")
    aic_values_adl = []
    p_q_vals = []
    models_adl = []
    for p in range(1, 5):
        for q in range(1, 5):
            model = ARDL(endog=un_series, lags=p, exog=gdp_df, order=q, trend='c')
            results = model.fit()
            aic_values_adl.append(results.aic)
            models_adl.append(results)
            p_q_vals.append([p, q])

    best_index = np.argmin(aic_values_adl)
    best_p_adl, best_q_adl = p_q_vals[best_index][0], p_q_vals[best_index][1]
    best_model_adl = models_adl[best_index]
    print(f"\nSelected ADL model for Unemployment: ADL({best_p_adl},{best_q_adl})")
    print(f"ADL({best_p_adl}, {best_q_adl}) AIC: {aic_values_adl[best_index]:.2f}\n")
    print(f"{best_model_adl.summary()}\n")

    return best_model_ar, best_model_adl


def part2_3(alpha, phi, beta, X_bar):
    first_frac = alpha / (1 - np.sum(phi))
    second_frac = (np.sum(beta)) / (1 - np.sum(phi))
    Y_bar = first_frac + second_frac * X_bar

    print(
        f"The long-run equilibruim unemployment rate for a fixed GDP growth rate of {X_bar}% is approximately {Y_bar:.2f}%.\n")


def part2_4(ar_model, adl_model, horizon=80):
    print("\n--- PART 2.4: Impulse Response Functions ---")
    # --- 1. Extract parameters ---
    ar_params = ar_model.params
    p_gdp = ar_model.model.order[0]
    phi_gdp = ar_params[1:1 + p_gdp].values

    adl_params = adl_model.params
    p_un = len(adl_model.ar_lags)

    # number of betas = total params - ar params
    num_beta_coeffs = len(adl_params) - 1 - p_un
    # The order 'q' is the number of beta coefficients minus one (since it includes beta_0).
    q_un_lags = num_beta_coeffs - 1

    phi_un = adl_params[1:1 + p_un].values
    beta_un = adl_params[1 + p_un:].values

    # --- 2. Initialize derivative arrays ---
    d_gdp = np.zeros(horizon)
    d_un = np.zeros(horizon)

    # --- 3. Recursive computation of derivatives ---
    d_gdp[0] = 1.0

    # In the code, d_un[h] represents ∂Y_{k+h}/∂u_k

    # Initial Condition (before the loop)
    if len(beta_un) > 0:
        d_un[0] = beta_un[0] * d_gdp[0]

    for h in range(1, horizon):
        # GDP response (AR process)
        for i in range(p_gdp):
            if h - (i + 1) >= 0:
                d_gdp[h] += phi_gdp[i] * d_gdp[h - (i + 1)]

        # Unemployment response (ADL process)
        # AR part
        for i in range(p_un):
            if h - (i + 1) >= 0:
                d_un[h] += phi_un[i] * d_un[h - (i + 1)]
        # DL part
        for j in range(q_un_lags + 1):
            if h - j >= 0:
                d_un[h] += beta_un[j] * d_gdp[h - j]

    # --- 4. Calculate IRFs for good and bad shocks ---
    s_good = 2.0
    s_bad = -2.0
    irf_gdp_good = d_gdp * s_good
    irf_gdp_bad = d_gdp * s_bad
    irf_un_good = d_un * s_good
    irf_un_bad = d_un * s_bad

    # --- 5. Plotting ---
    print("Plotting IRFs...")
    time_axis = np.arange(horizon)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Impulse Response Functions to a Shock in GDP Growth', fontsize=16)

    # GDP IRF plot
    axs[0].plot(time_axis, irf_gdp_good, label='Response to Positive Shock (+2%)', color='green', marker='o',
                markersize=3, alpha=0.7)
    axs[0].plot(time_axis, irf_gdp_bad, label='Response to Negative Shock (-2%)', color='red', marker='x', markersize=3,
                alpha=0.7)
    axs[0].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[0].set_title('Dynamic Response of GDP Growth Rate')
    axs[0].set_ylabel('Change in GDP Growth (%)')
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].legend()

    # Unemployment IRF plot
    axs[1].plot(time_axis, irf_un_good, label='Response to Positive GDP Shock', color='green', marker='o', markersize=3,
                alpha=0.7)
    axs[1].plot(time_axis, irf_un_bad, label='Response to Negative GDP Shock', color='red', marker='x', markersize=3,
                alpha=0.7)
    axs[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
    axs[1].set_title('Dynamic Response of Unemployment Rate')
    axs[1].set_xlabel('Quarters after Shock')
    axs[1].set_ylabel('Change in Unemployment Rate (%)')
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":

    part1_1()
    best_model_p1, best_p_p1 = part1_2()
    part1_3_and_4(best_model_p1, best_p_p1)

    gdp_df, un_series = dataloader2()

    best_model_ar, best_model_adl = part2_1(gdp_df, un_series)

    p_un_selected = len(best_model_adl.ar_lags)
    params_adl_selected = best_model_adl.params
    alpha_hat = params_adl_selected['const']
    phi_hat = params_adl_selected[1:1 + p_un_selected].values
    beta_hat = params_adl_selected[1 + p_un_selected:].values
    part2_3(alpha_hat, phi_hat, beta_hat, X_bar=2.0)

    part2_4(best_model_ar, best_model_adl)
