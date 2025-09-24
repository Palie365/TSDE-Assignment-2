import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
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
    # ts plot
    plt.figure(figsize=(10, 4))
    plt.plot(data1, color='#4682b4', linewidth=1.5)
    plt.title("Dutch GDP Quarterly Growth Rate")
    plt.xlabel("Time") ### change index in figure, should show years
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

def dataloader2():
    GDP_df = data2_df.copy()
    GDP_df.drop(columns=["UN_RATE"], axis=1, inplace=True)
    GDP_data = GDP_df[['GDP_QGR']].values

    UN_df = data2_df.copy()
    UN_df.drop(columns=["GDP_QGR"], axis=1, inplace=True)
    UN_data = UN_df[['UN_RATE']].values.flatten()
    
    return GDP_data, UN_data
    
def part2_1(GDP_data, UN_data):        
    ### ts plot of GDP and UN
    plt.figure(figsize=(10, 4))
    plt.plot(GDP_data, color='tab:blue', linewidth=1.5, label='GDP Growth Rate')
    plt.plot(UN_data, color='tab:orange', linewidth=1.5, label='Unemployment Rate')
    plt.title("Dutch GDP Quarterly Growth Rate and Unemployment Rate")
    plt.xlabel("T")                          ### change index in figure, should show years
    plt.ylabel("GDP Growth and Unemployment (%)")
    plt.grid(True, alpha=0.1)
    plt.legend()
    #plt.savefig('Data_part2.png')
    plt.show()
    
    ### estimate AR() model for GDP   
    aic_values = []
    models = {}

    # AR(p) models with p = 1, 2, 3, 4
    for p in range(1, 5):
        model = sm.tsa.ARIMA(GDP_data, order=(p, 0, 0), trend="c")
        results = model.fit()
        aic_values.append(results.aic)
        models[p] = results
        print(f"AR({p}) AIC: {results.aic:.2f}")

    # Select best p by minimum AIC
    best_p = np.argmin(aic_values) + 1
    best_model = models[best_p]

    print(f"\nSelected model: AR({best_p})")
    print(best_model.summary())
    
    ### estimate ADL() model for UN, where GDP is exogenous variable Xt
    aic_values = []
    p_q_vals = []
    models = []
    
    # ADL(p, q) models with p=1,2,3,4 and q=0,1,2,3,4
    for p in range(1,5):
        for q in range(1,5):
            model = ARDL(endog=UN_data, lags=p, exog=GDP_data, order=q, trend='c')    
            results = model.fit()
            aic_values.append(results.aic)
            models.append(results)
            p_q_vals.append([p, q])
    
    best_index = np.argmin(aic_values) 
    best_p, best_q = p_q_vals[best_index][0], p_q_vals[best_index][1]
    best_model = models[best_index]
    
    print(f"\nSelected model: ADL({best_p},{best_q})")
    print(f"ADL({best_p}, {best_q}) AIC: {aic_values[best_index]}\n")
    print(f"{best_model.summary()}\n")
    
    return best_model
    
def part2_3(alpha, phi, beta, X_bar):
    first_frac = alpha/(1 - np.sum(phi))
    second_frac = (np.sum(beta))/(1 - np.sum(phi))
    Y_bar = first_frac + second_frac * X_bar  
     
    print(f"The long-run equilibruim unemployment rate for a fixed GDP growth rate of {X_bar * 100}% is approximately {Y_bar:.2f}%.\n")

if __name__ == "__main__":
    part1_1()
    best_model, best_p = part1_2()

    GDP_data, UN_data = dataloader2()
    best_model = part2_1(GDP_data, UN_data)

    alpha_hat, phi_hat, beta_hat = best_model.params[0], np.array([best_model.params[1], best_model.params[2], best_model.params[3]]), np.array([best_model.params[4], best_model.params[5]])
    part2_3(alpha_hat, phi_hat, beta_hat, 0.02)                 ### we could make a figure of Y_bar as a function of X_bar if we want. could be a nice addition
