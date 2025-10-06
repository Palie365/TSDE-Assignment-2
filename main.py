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

    # --- CORRECTION ---
    # The 'const' from sm.tsa.ARIMA is the process mean (μ), not the intercept (c).
    mean_hat = best_model.params[0]
    phi_hat = best_model.params[1]

    # correction to compute the real intercepts
    intercept_hat = mean_hat * (1 - phi_hat)

    # forecasts
    forecasts = []
    X_T = data1[T1 - 1]
    for _ in range(h):
        next_val = intercept_hat + phi_hat * X_T
        forecasts.append(next_val)
        X_T = next_val
    forecasts = np.array(forecasts)

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


def part2_4(gdp_ar_model, unemployment_adl_model, gdp_data, un_data, horizon=70):
    # get the last values from the data to use as the starting point for the plots
    gdp_start_value = gdp_data.iloc[-1].values[0]
    unemployment_start_value = un_data.iloc[-1]
    print(f"Using origins: GDP Growth = {gdp_start_value:.2f}, Unemployment = {unemployment_start_value:.2f}")

    # correction to compute the real intercepts
    gdp_equilibrium = gdp_ar_model.params['const']
    adl_params = unemployment_adl_model.params
    unemployment_const = adl_params['const']
    unemployment_p = len(unemployment_adl_model.ar_lags)
    unemployment_ar_coeffs = adl_params[1: 1 + unemployment_p].values
    unemployment_dl_coeffs = adl_params[1 + unemployment_p:].values
    unemployment_equilibrium = (unemployment_const + np.sum(unemployment_dl_coeffs) * gdp_equilibrium) / (
                1 - np.sum(unemployment_ar_coeffs))
    print(
        f"Plotting convergence to equilibrium: GDP Mean = {gdp_equilibrium:.2f}, Unemployment Mean = {unemployment_equilibrium:.2f}")

    # get coefficients from the GDP model
    gdp_ar_coeffs = gdp_ar_model.params[1:].values
    gdp_p = gdp_ar_model.model.order[0]

    # get coefficients from the Unemployment ADL model
    num_beta_coeffs = len(adl_params) - 1 - unemployment_p
    unemployment_q = num_beta_coeffs - 1

    # create empty arrays to store the results of the derivative calculations
    gdp_derivatives = np.zeros(horizon)
    unemployment_derivatives = np.zeros(horizon)

    # the first shock has a size of 1
    gdp_derivatives[0] = 1.0
    unemployment_derivatives[0] = unemployment_dl_coeffs[0] * gdp_derivatives[0]

    # loop through time to calculate the effect of the shock step by step
    for h in range(1, horizon):
        for i in range(gdp_p):
            if h - (i + 1) >= 0:
                gdp_derivatives[h] += gdp_ar_coeffs[i] * gdp_derivatives[h - (i + 1)]
        for i in range(unemployment_p):
            if h - (i + 1) >= 0:
                unemployment_derivatives[h] += unemployment_ar_coeffs[i] * unemployment_derivatives[h - (i + 1)]
        for j in range(unemployment_q + 1):
            if h - j >= 0:
                unemployment_derivatives[h] += unemployment_dl_coeffs[j] * gdp_derivatives[h - j]

    # define the shock sizes from the assignment
    positive_shock_size = 2.0
    negative_shock_size = -2.0

    # calculate the final paths by adding the starting value to the shock effect
    gdp_path_positive_shock = gdp_start_value + (gdp_derivatives * positive_shock_size)
    gdp_path_negative_shock = gdp_start_value + (gdp_derivatives * negative_shock_size)
    unemployment_path_positive_shock = unemployment_start_value + (unemployment_derivatives * positive_shock_size)
    unemployment_path_negative_shock = unemployment_start_value + (unemployment_derivatives * negative_shock_size)

    # origin line visual
    pre_shock_periods = 5
    pre_shock_time_axis = np.arange(-pre_shock_periods, 0)
    post_shock_time_axis = np.arange(horizon)
    time_axis = np.concatenate([pre_shock_time_axis, post_shock_time_axis])
    pre_shock_gdp_path = np.full(pre_shock_periods, gdp_start_value)
    pre_shock_unemployment_path = np.full(pre_shock_periods, unemployment_start_value)
    full_gdp_path_positive = np.concatenate([pre_shock_gdp_path, gdp_path_positive_shock])
    full_gdp_path_negative = np.concatenate([pre_shock_gdp_path, gdp_path_negative_shock])
    full_unemployment_path_positive = np.concatenate([pre_shock_unemployment_path, unemployment_path_positive_shock])
    full_unemployment_path_negative = np.concatenate([pre_shock_unemployment_path, unemployment_path_negative_shock])

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # top plot for GDP
    axs[0].plot(time_axis, full_gdp_path_positive, label='Response to Positive Shock (+2%)', color='green', marker='o',
                markersize=3, alpha=0.7)
    axs[0].plot(time_axis, full_gdp_path_negative, label='Response to Negative Shock (-2%)', color='red', marker='x',
                markersize=3, alpha=0.7)
    axs[0].set_title('IRF GDP')
    axs[0].set_ylabel('GDP Growth Rate (%)')
    axs[0].grid(True, alpha=0.4)
    axs[0].legend()

    # bottom plot for unemployment
    axs[1].plot(time_axis, full_unemployment_path_positive, label='Response to Positive GDP Shock (+2%)', color='green',
                marker='o', markersize=3, alpha=0.7)
    axs[1].plot(time_axis, full_unemployment_path_negative, label='Response to Negative GDP Shock (-2%)', color='red',
                marker='x', markersize=3, alpha=0.7)

    axs[1].set_title('IRF Unemployment rate')
    axs[1].set_xlabel('Quarters after Shock')
    axs[1].set_ylabel('Unemployment Rate (%)')
    axs[1].grid(True, alpha=0.4)
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

    part2_4(gdp_ar_model=best_model_ar, unemployment_adl_model=best_model_adl, gdp_data=gdp_df, un_data=un_series)

