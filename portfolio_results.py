# Get portfolio data

import pandas as pd
from datetime import datetime
import statsmodels.api as sm
pd.set_option('display.width', 1000)

def convert_to_monthly(date):

    year = date.year
    month = int(date.strftime('%d'))  # Assuming the day value represents the month
    return pd.Timestamp(f"{year}-{month:02d}-01")

recession_forecasts = pd.read_csv(r'val_f1\01_model_lag20_est256_depth8_predictions.csv')

ff_data = pd.read_csv('ff_data.csv')
ff_data.iloc[:, 1:] = ff_data.iloc[:, 1:]/100

def join_recession_and_ff(recession_df, ff_df):

    recession_df['date'] = pd.to_datetime(recession_df['date'])
    ff_df['date'] = pd.to_datetime(ff_df['date'], format='%d/%m/%Y')
    merged_df = pd.merge(recession_df, ff_df, on='date', how='inner')
    
    return merged_df

result_df = join_recession_and_ff(recession_forecasts, ff_data)

def simulate_strategy_and_regress(data):

    data["date"] = pd.to_datetime(data["date"])
    data["Mkt"] = data["Mkt-RF"] + data['RF']
    data["portfolio_return"] = data["Mkt"].where(data["binary_prediction"] == 0, data["RF"])
    data["excess_return"] = data["portfolio_return"] - data["RF"]
    
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = data[factors]
    X = sm.add_constant(X)
     
    y = data["excess_return"]
    model = sm.OLS(y, X).fit()
    
    return data, model


result_df, model = simulate_strategy_and_regress(result_df)
print(model.summary())


import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter
import matplotlib.patches as mpatches

def plot_cumulative_returns(data):

    data["portfolio_cum_return"] = (1 + data["portfolio_return"]).cumprod()
    data["market_cum_return"] = (1 + data["Mkt"]).cumprod()
    
    data["date"] = pd.to_datetime(data["date"])
    
    plt.figure(figsize=(14, 8))
    
    plt.plot(data["date"], data["portfolio_cum_return"], label="Portfolio", linewidth=2)
    plt.plot(data["date"], data["market_cum_return"], label="Market", linewidth=2, linestyle='dashed')
    
    in_recession = False
    start_date = None
    
    for i in range(len(data)):
        if data["binary_prediction"].iloc[i] == 1 and not in_recession:
            start_date = data["date"].iloc[i]
            in_recession = True
        elif data["binary_prediction"].iloc[i] == 0 and in_recession:
            plt.axvspan(start_date, data["date"].iloc[i], color='gray', alpha=0.3)
            in_recession = False
    
    if in_recession:
        plt.axvspan(start_date, data["date"].iloc[-1], color='gray', alpha=0.3)
    
    recession_patch = mpatches.Patch(color='gray', alpha=0.3, label='Forecasted Recession')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator(3))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(recession_patch)
    plt.legend(handles=handles)
    
    plt.grid()
    plt.show()

plot_cumulative_returns(result_df)


def print_recession_ranges(result_df):

    result_df = result_df.copy()
    result_df['date'] = pd.to_datetime(result_df['date'])
    
    changes = result_df['binary_prediction'].diff().fillna(0)
    
    start_indices = result_df[changes == 1].index.tolist()
    end_indices = result_df[changes == -1].index.tolist()
    
    if result_df['binary_prediction'].iloc[-1] == 1:
        end_indices.append(len(result_df) - 1)
    
    print("\nRecession Periods (±2 months):")
    print("=" * 80)
    
    for start_idx in start_indices:
        end_idx = next((x for x in end_indices if x > start_idx), None)
        
        if end_idx is None:
            continue
            
        range_start = max(0, start_idx - 2)
        range_end = min(len(result_df) - 1, end_idx + 2)
        
        range_data = result_df.iloc[range_start:range_end + 1]
        
        print(f"\nPeriod from {range_data['date'].iloc[0].strftime('%Y-%m')} "
              f"to {range_data['date'].iloc[-1].strftime('%Y-%m')}:")
        print("-" * 80)
        
        print(range_data[['date', 'predicted_probability', 'binary_prediction', 'Mkt-RF', 'Mkt', 'RF', 'portfolio_return']])
        print("\n")

print_recession_ranges(result_df)


# Regression for a given time period (train, validation or test)

def simulate_strategy_and_regress(data, period='train'):

    data["date"] = pd.to_datetime(data["date"])
    data["Mkt"] = data["Mkt-RF"] + data['RF']
    
    training_start = '1976-07-01'
    training_end = '1998-01-01'
    validation_start = '1998-01-02'
    validation_end = '2004-01-01'
    test_start = '2004-01-02'
    
    if period == 'train':
        period_data = data[(data["date"] >= training_start) & 
                          (data["date"] <= training_end)].copy()
        period_label = f"{training_start} to {training_end}"
    elif period == 'validation':
        period_data = data[(data["date"] > training_end) & 
                          (data["date"] <= validation_end)].copy()
        period_label = f"{validation_start} to {validation_end}"
    elif period == 'test':
        period_data = data[data["date"] > validation_end].copy()
        period_label = f"{test_start} to present"
    else:
        raise ValueError("Period must be 'train', 'validation', or 'test'")
    
    period_data["portfolio_return"] = period_data["Mkt"].where(
        period_data["binary_prediction"] == 0, 
        period_data["RF"]
    )
    
    period_data["excess_return"] = period_data["portfolio_return"] - period_data["RF"]
    
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    X = period_data[factors]
    X = sm.add_constant(X)
    
    y = period_data["excess_return"]
    model = sm.OLS(y, X).fit()
    
    period_data["portfolio_cum_return"] = (1 + period_data["portfolio_return"]).cumprod()
    period_data["market_cum_return"] = (1 + period_data["Mkt"]).cumprod()
    
    plt.figure(figsize=(14, 8))
    
    plt.plot(period_data["date"], period_data["portfolio_cum_return"], label="Portfolio", linewidth=2)
    plt.plot(period_data["date"], period_data["market_cum_return"], label="Market", linewidth=2, linestyle='dashed')
    
    in_recession = False
    start_date = None
    
    for i in range(len(period_data)):
        if period_data["binary_prediction"].iloc[i] == 1 and not in_recession:
            start_date = period_data["date"].iloc[i]
            in_recession = True
        elif period_data["binary_prediction"].iloc[i] == 0 and in_recession:
            plt.axvspan(start_date, period_data["date"].iloc[i], color='gray', alpha=0.3)
            in_recession = False
    
    if in_recession:
        plt.axvspan(start_date, period_data["date"].iloc[-1], color='gray', alpha=0.3)
    
    recession_patch = mpatches.Patch(color='gray', alpha=0.3, label='Forecasted Recession')
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator(3))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title(f"Cumulative Returns for {period.capitalize()} Period ({period_label})")
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(recession_patch)
    plt.legend(handles=handles)
    
    plt.grid()
    plt.show()
    
    results = {
        period: {
            'model': model,
            'n_obs': len(period_data),
            'period': period_label}}
    
    return results


print("Train period results:")
train_results = simulate_strategy_and_regress(result_df, 'train')
train_model = train_results['train']['model']
print(train_model.summary())
print("")


print("Validation period results:")
val_results = simulate_strategy_and_regress(result_df, 'validation')
val_model = val_results['validation']['model']
print(val_model.summary())
print("")


print("Test period results:")
test_results = simulate_strategy_and_regress(result_df, 'test')
test_model = test_results['test']['model']
print(test_model.summary())
