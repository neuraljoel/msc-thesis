# Set seeds for reproducibility


import numpy as np
np.random.seed(42)


# Get data for random forests


import pandas as pd
from fredapi import Fred
from datetime import datetime


def load_recession_indicators(api_key, start_date='1976-06-01', end_date='2024-12-01'):
    fred = Fred(api_key=api_key)
    
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'INDPRO': 'Industrial Production',
        'PAYEMS': 'Nonfarm Payrolls',
        'CFNAI': 'Chicago Fed National Activity Index',
        'T10Y2Y': '10Y-2Y Treasury Spread',
    }
    
    data = pd.DataFrame()
    invalid_series = []
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    for series_id, description in indicators.items():
        series = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
            frequency='m',
            aggregation_method='eop'
        )
        
        if series.empty:
            invalid_series.append(f"{series_id}: No data available")
            continue
            
        series_start = series.index.min()
        series_end = series.index.max()
        
        # Modified validation for CFNAI
        if series_id == 'CFNAI':
            if series_end < end_dt - pd.DateOffset(months=1):
                invalid_series.append(f"{series_id}: Data too old, ends at {series_end.strftime('%Y-%m-%d')}")
                continue
        else:
            # Original validation for other indicators
            if series_start > start_dt:
                invalid_series.append(f"{series_id}: Starts at {series_start.strftime('%Y-%m-%d')}")
                continue
            if series_end < end_dt:
                invalid_series.append(f"{series_id}: Ends at {series_end.strftime('%Y-%m-%d')}")
                continue
            
        if series.isnull().any():
            invalid_series.append(f"{series_id}: Contains missing values")
            continue
            
        data[series_id] = series
    
    if invalid_series:
        print("Warning: The following series failed validation:")
        for msg in invalid_series:
            print(f"- {msg}")
    
    if data.empty:
        raise ValueError("No valid series found for the specified date range")
    
    return data

df = load_recession_indicators('API_KEY_HERE')

def convert_to_monthly(date):
    """
    Convert a date to a monthly timestamp where the day represents the month.
    
    Args:
        date (pd.Timestamp): A pandas datetime object.
        
    Returns:
        pd.Timestamp: A timestamp with the month derived from the day and the day set to 01.
    """
    year = date.year
    month = int(date.strftime('%d'))  # Assuming the day value represents the month
    return pd.Timestamp(f"{year}-{month:02d}-01")




import os

sp_500 = pd.read_csv('SP.csv')


def handle_data():

    recessions = pd.read_csv('USRECDM.csv')
    rp = pd.read_csv('RP.csv')
    sp_500 = pd.read_csv('SP.csv')

    all_data = pd.merge(recessions, sp_500, on="DATE", how="left")
    all_data = pd.merge(all_data, rp, on="DATE", how="left")
    
    all_data['RP'] = all_data['RP']/100
    
    #all_data.dropna(inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    
    all_data['DATE'] = pd.to_datetime(all_data['DATE']).apply(convert_to_monthly)

    return all_data

all_data = handle_data()

df.reset_index(inplace=True)
df = df.rename(columns={'index': 'DATE'})
all_data = pd.merge(all_data[['DATE', 'USRECDM', 'RP', 'SP_INDEX']], df, on='DATE', how='outer')
all_data['UNRATE'] = all_data['UNRATE']/100


used_data = all_data[['DATE',
                       'UNRATE',
                       'INDPRO',
                       'PAYEMS',
                       'T10Y2Y',
                       'USRECDM',
                       'SP_INDEX',
                       'RP',
                       'CFNAI']]


# Proper forecasting for February 2025

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import itertools
import os
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

# Feature names
features = ['UNRATE', 'INDPRO', 'T10Y2Y', 'SP_INDEX', 'RP', 'CFNAI']
lag_3_features = ['RP', 'CFNAI']
pct_change_features = ['UNRATE', 'INDPRO']

# Data preparation function

def create_lagged_features(data, t10y2y_lag):
    """Create lagged features and combinations with T10Y2Y."""
    required_cols = features + ['USRECDM', 'DATE']
    data = data[required_cols].copy()
    
    data.index = pd.to_datetime(data['DATE'])
    data = data.drop('DATE', axis=1)
    lagged_data = data.copy()
    
    # Second lag for all features except lag_3_features
    for feature in features:
        if feature not in lag_3_features:
            lagged_data[f'{feature}_l2'] = lagged_data[feature].shift(2)
            lagged_data[f'{feature}_l2_rolling_mean_6m'] = lagged_data[f'{feature}_l2'].rolling(window=6).mean()
    
    # Second lag for specified features
    for feature in lag_3_features:
        lagged_data[f'{feature}_l3'] = lagged_data[feature].shift(3)
        lagged_data[f'{feature}_l3_rolling_mean_6m'] = lagged_data[f'{feature}_l3'].rolling(window=6).mean()
    
    # Percentage changes
    for feature in pct_change_features:
        lagged_data[f'{feature}_pct_l2'] = lagged_data[feature].pct_change(periods=1).shift(2)
        lagged_data[f'{feature}_pct_l2_rolling_mean_6m'] = lagged_data[f'{feature}_pct_l2'].rolling(window=6).mean()
    
    # Additional lag for T10Y2Y
    lagged_data[f'T10Y2Y_l{t10y2y_lag}'] = lagged_data['T10Y2Y'].shift(t10y2y_lag)
    lagged_data[f'T10Y2Y_l{t10y2y_lag}_rolling_mean_6m'] = lagged_data[f'T10Y2Y_l{t10y2y_lag}'].rolling(window=6).mean()
    
    # Drop original features
    lagged_data = lagged_data.drop(columns=features)
    
    return lagged_data


def create_future_features(last_data, feature_cols, t10y2y_lag):
    """Create features for future prediction matching the historical feature creation pattern."""
    extended_data = last_data.copy()
    future_date = pd.Timestamp('2025-02-01')
    extended_data.loc[future_date] = None
    
    # Create shifted features
    for feature in features:
        if feature not in lag_3_features:
            extended_data[f'{feature}_l2'] = extended_data[feature].shift(2)
            extended_data[f'{feature}_l2_rolling_mean_6m'] = extended_data[feature].shift(2).rolling(window=6).mean()
    
    for feature in lag_3_features:
        extended_data[f'{feature}_l3'] = extended_data[feature].shift(3)
        extended_data[f'{feature}_l3_rolling_mean_6m'] = extended_data[feature].shift(3).rolling(window=6).mean()
    
    for feature in pct_change_features:
        extended_data[f'{feature}_pct_l2'] = extended_data[feature].pct_change(periods=1).shift(2)
        extended_data[f'{feature}_pct_l2_rolling_mean_6m'] = extended_data[feature].pct_change(periods=1).shift(2).rolling(window=6).mean()
    
    extended_data[f'T10Y2Y_l{t10y2y_lag}'] = extended_data['T10Y2Y'].shift(t10y2y_lag)
    extended_data[f'T10Y2Y_l{t10y2y_lag}_rolling_mean_6m'] = extended_data['T10Y2Y'].shift(t10y2y_lag).rolling(window=6).mean()
    
    # Extract only the future date's features
    future_features = extended_data.loc[[future_date], feature_cols]
    
    return future_features



# Save all plots and CSV outputs for a given model

def save_results(model_results, folder_name, dates, probabilities, true_values, 
                 feature_importances, feature_names, file_prefix, 
                 future_dates=None, future_probabilities=None):

    os.makedirs(folder_name, exist_ok=True)
    mpl_dir = os.path.join(folder_name, 'mpl_plots')
    plt_dir = os.path.join(folder_name, 'plt_plots')
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    train_end = pd.Timestamp('1998-01-01')
    val_end = pd.Timestamp('2004-01-01')
    
    # Calculate binary predictions
    binary_predictions = (probabilities >= 0.5).astype(int)

    fig_ts_mpl = plt.figure(figsize=(12, 6))
    plt.plot(dates, probabilities, color='red', alpha=0.7, label='Predicted Probabilities')
    plt.plot(dates, true_values, color='blue', alpha=0.7, label='True Values')
    plt.plot(dates, binary_predictions, color='#006400', alpha=0.7, label='Binary Prediction')
    
    if future_dates is not None and future_probabilities is not None:
    # Orange line to connect
        connection_dates = [dates[-1], future_dates[0]]
        connection_probas = [probabilities[-1], future_probabilities[0]]
        plt.plot(connection_dates, connection_probas, color='orange', linewidth=2.5, label='Future Prediction')
        plt.scatter(future_dates[0], future_probabilities[0], color='orange', s=50, zorder=5)
    
    # Vertical lines for train/val/test
    plt.axvline(x=train_end, color='gray', linestyle='--', alpha=0.7, label='Train/Val Split')
    plt.axvline(x=val_end, color='black', linestyle='--', alpha=0.7, label='Val/Test Split')
    
    model_params = (f"T10Y2Y Lag: {model_results['t10y2y_lag']}, "
                    f"n_estimators: {model_results['n_estimators']}, "
                    f"max_depth: {model_results['max_depth']}")
    metrics_str = (f"Train f1: {float(model_results['train_f1']):.5f}\n"
                   f"Val f1: {float(model_results['val_f1']):.5f}\n"
                   f"Test f1: {float(model_results['test_f1']):.5f}")
    
    plt.title(f"Random Forest Recession Predictions\n{model_params}\n{metrics_str}")
    plt.xlabel('Date')
    plt.ylabel('Probability [0,1]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{mpl_dir}/{file_prefix}_timeseries.png', bbox_inches='tight')
    plt.close()

    fig_fi_mpl = plt.figure(figsize=(12, 6))
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=True)
    
    plt.barh(np.arange(len(importance_df)), importance_df['importance'])
    plt.yticks(np.arange(len(importance_df)), importance_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{mpl_dir}/{file_prefix}_feature_importance.png')
    plt.close()

    # Plotly figure for interactive visualization
    fig_ts_plt = go.Figure()
    fig_ts_plt.add_trace(
        go.Scatter(
            x=dates, y=true_values, name='True Values',
            line=dict(color='blue', width=2)
        )
    )
    fig_ts_plt.add_trace(
        go.Scatter(
            x=dates, y=binary_predictions, name='Binary Prediction',
            line=dict(color='#006400', width=2)
        )
    )

    if future_dates is not None and future_probabilities is not None:
        all_dates = np.concatenate([dates, future_dates])
        all_probas = np.concatenate([probabilities, future_probabilities])
        fig_ts_plt.add_trace(
            go.Scatter(
                x=all_dates, 
                y=all_probas,
                name='Predictions',
                line=dict(color='red', width=2)
            )
        )
        # Marker for future prediction
        fig_ts_plt.add_trace(
            go.Scatter(
                x=[future_dates[0]], 
                y=[future_probabilities[0]],
                mode='markers',
                marker=dict(color='orange', size=12),
                name='Feb 2025 Prediction'
            )
        )
    else:
        fig_ts_plt.add_trace(
            go.Scatter(
                x=dates, y=probabilities, name='Predicted Probabilities',
                line=dict(color='red', width=2)
            )
        )

    fig_ts_plt.add_shape(
        type="line", x0=train_end, x1=train_end, y0=0, y1=1,
        line=dict(dash="dash", color="gray"), name="Train/Val Split"
    )
    fig_ts_plt.add_shape(
        type="line", x0=val_end, x1=val_end, y0=0, y1=1,
        line=dict(dash="dash", color="black"), name="Val/Test Split"
    )
    
    fig_ts_plt.add_annotation(
        x=train_end, y=1.05, text="Train/Val Split",
        showarrow=False, yref="paper"
    )
    fig_ts_plt.add_annotation(
        x=val_end, y=1.1, text="Val/Test Split",
        showarrow=False, yref="paper"
    )

    fig_ts_plt.update_layout(
        title=f"Random Forest Recession Predictions<br>{model_params}<br>{metrics_str}",
        xaxis_title='Date',
        yaxis_title='Probability [0,1]',
        height=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig_ts_plt.write_html(f'{plt_dir}/{file_prefix}_timeseries.html')

    # -- Plotly Feature Importance --
    fig_fi_plt = go.Figure()
    fig_fi_plt.add_trace(
        go.Bar(
            y=importance_df['feature'],
            x=importance_df['importance'],
            orientation='h',
            showlegend=False
        )
    )
    fig_fi_plt.update_layout(
        title='Random Forest Feature Importance',
        xaxis_title='Feature Importance',
        height=600,
        margin=dict(l=200)
    )
    fig_fi_plt.write_html(f'{plt_dir}/{file_prefix}_feature_importance.html')

    # -- Save predictions CSV --
    predictions_df = pd.DataFrame({
        'date': dates,
        'predicted_probability': probabilities,
        'binary_prediction': binary_predictions,
        'true_value': true_values
    })
    if future_dates is not None and future_probabilities is not None:
        future_binary_predictions = (future_probabilities >= 0.5).astype(int)
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_probability': future_probabilities,
            'binary_prediction': future_binary_predictions,
            'true_value': [None] * len(future_dates)
        })
        predictions_df = pd.concat([predictions_df, future_df])
    
    predictions_df.to_csv(f'{folder_name}/{file_prefix}_predictions.csv', index=False)



# Model execution function

def run_model(used_data):
    
    train_end = pd.Timestamp('1998-01-01')
    val_end = pd.Timestamp('2004-01-01')
    
    # -- Define a clear param grid --
    param_grid = {
        't10y2y_lags': [4, 20],
        'n_estimators': [128, 256, 512],
        'max_depth': [2, 8, None]
    }
    
    all_results = []
    val_f1_models = []
    test_f1_models = []
    NUM_MODELS_TO_SAVE = 20
    
    for t10y2y_lag in param_grid['t10y2y_lags']:
        print(f"\nProcessing T10Y2Y lag: {t10y2y_lag} months")
        
        # Create lagged features
        lagged_data = create_lagged_features(used_data, t10y2y_lag)
        lagged_data = lagged_data.dropna()
        
        # Split data
        train_data = lagged_data[lagged_data.index <= train_end]
        val_data = lagged_data[(lagged_data.index > train_end) & (lagged_data.index <= val_end)]
        test_data = lagged_data[lagged_data.index > val_end]
        
        feature_cols = [col for col in lagged_data.columns if col != 'USRECDM']
        X_train = train_data[feature_cols]
        y_train = train_data['USRECDM']
        X_val = val_data[feature_cols]
        y_val = val_data['USRECDM']
        X_test = test_data[feature_cols]
        y_test = test_data['USRECDM']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Iterate over n_estimators and max_depth
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid['max_depth']:
                
                # Train the model
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(X_train_scaled, y_train)
                
                # Generate predictions
                train_pred_proba = rf.predict_proba(X_train_scaled)[:, 1]
                val_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
                test_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
                
                train_pred = (train_pred_proba >= 0.5).astype(int)
                val_pred = (val_pred_proba >= 0.5).astype(int)
                test_pred = (test_pred_proba >= 0.5).astype(int)
                
                # Compute metrics
                metrics_dict = {
                    'train_f1': f1_score(y_train, train_pred, zero_division=0),
                    'val_f1': f1_score(y_val, val_pred, zero_division=0),
                    'test_f1': f1_score(y_test, test_pred, zero_division=0),
                    't10y2y_lag': t10y2y_lag,
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
                all_results.append(metrics_dict)
                
                # Generate future prediction for 2025-02-01
                future_data = create_future_features(used_data, feature_cols, t10y2y_lag)
                if not future_data.empty and not future_data.isna().all().all():
                    future_scaled = scaler.transform(future_data)
                    future_proba = rf.predict_proba(future_scaled)[:, 1]
                    future_dates = pd.date_range(start='2025-02-01', end='2025-02-01', freq='MS')
                else:
                    future_dates = None
                    future_proba = None
                
                # Store all data needed for saving/plotting
                model_data = {
                    'metrics': metrics_dict,
                    'dates': np.concatenate([train_data.index.values, val_data.index.values, test_data.index.values]),
                    'proba': np.concatenate([train_pred_proba, val_pred_proba, test_pred_proba]),
                    'true': np.concatenate([y_train.values, y_val.values, y_test.values]),
                    'importances': rf.feature_importances_,
                    'features': feature_cols,
                    'prefix': f'model_lag{t10y2y_lag}_est{n_estimators}_depth{max_depth}',
                    'future_dates': future_dates,
                    'future_proba': future_proba
                }

                # Keep track of top models by val_f1
                val_f1_models.append(model_data)
                val_f1_models.sort(key=lambda x: x['metrics']['val_f1'], reverse=True)
                val_f1_models = val_f1_models[:NUM_MODELS_TO_SAVE]

                # Keep track of top models by test_f1
                test_f1_models.append(model_data)
                test_f1_models.sort(key=lambda x: x['metrics']['test_f1'], reverse=True)
                test_f1_models = test_f1_models[:NUM_MODELS_TO_SAVE]

    # Save all results to CSV
    pd.DataFrame(all_results).to_csv('all_model_results.csv', index=False)

    # Save top validation-F1 models
    for i, model in enumerate(val_f1_models):
        save_results(
            model['metrics'],
            folder_name='val_f1',
            dates=model['dates'],
            probabilities=model['proba'],
            true_values=model['true'],
            feature_importances=model['importances'],
            feature_names=model['features'],
            file_prefix=f"{i+1:02d}_{model['prefix']}",
            future_dates=model['future_dates'],
            future_probabilities=model['future_proba']
        )

    # Save top test-F1 models
    for i, model in enumerate(test_f1_models):
        save_results(
            model['metrics'],
            folder_name='test_f1',
            dates=model['dates'],
            probabilities=model['proba'],
            true_values=model['true'],
            feature_importances=model['importances'],
            feature_names=model['features'],
            file_prefix=f"{i+1:02d}_{model['prefix']}",
            future_dates=model['future_dates'],
            future_probabilities=model['future_proba'])


# Run all models to select the best one in terms of F1 score
run_model(used_data)





# Code for running a specific model

def save_results(model_results, folder_name, dates, probabilities, true_values, 
                feature_importances, feature_names, file_prefix, 
                future_dates=None, future_probabilities=None):

    os.makedirs(folder_name, exist_ok=True)
    mpl_dir = os.path.join(folder_name, 'mpl_plots')
    plt_dir = os.path.join(folder_name, 'plt_plots')
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    binary_predictions = (probabilities >= 0.5).astype(int)

    fig_ts_mpl = plt.figure(figsize=(14, 8))
    plt.plot(dates, probabilities, color='red', alpha=0.7, label='Predicted Probabilities')
    plt.plot(dates, true_values, color='blue', alpha=0.7, label='True Values')
    plt.plot(dates, binary_predictions, color='#006400', alpha=0.7, label='Binary Prediction')
    
    if file_prefix == 'test' and future_dates is not None and future_probabilities is not None:
        connection_dates = [dates[-1], future_dates[0]]
        connection_probas = [probabilities[-1], future_probabilities[0]]
        plt.plot(connection_dates, connection_probas, color='orange', linewidth=2.5, label='Future Prediction')
        plt.scatter(future_dates[0], future_probabilities[0], color='orange', s=50, zorder=5)
    
    title_mapping = {
        'train': 'Training set results',
        'val': 'Validation set results',
        'test': 'Test set results'
    }
    
    metrics_str = f"F1-score: {float(model_results[f'{file_prefix}_f1']):.5f}"
    
    plt.xlabel('Date')
    plt.ylabel('Probability$\in [0,1]$')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{mpl_dir}/{file_prefix}_timeseries.png', bbox_inches='tight')
    plt.close()

    if file_prefix == 'test':
        plt.figure(figsize=(14, 10))
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=True)
        
        plt.barh(np.arange(len(importance_df)), importance_df['importance'])
        plt.yticks(np.arange(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        plt.savefig(f'{mpl_dir}/feature_importance.png')
        plt.close()

    fig_ts_plt = go.Figure()
    
    fig_ts_plt.add_trace(
        go.Scatter(
            x=dates, y=true_values, name='True Values',
            line=dict(color='blue', width=2)
        )
    )
    fig_ts_plt.add_trace(
        go.Scatter(
            x=dates, y=binary_predictions, name='Binary Prediction',
            line=dict(color='#006400', width=2)
        )
    )
    fig_ts_plt.add_trace(
        go.Scatter(
            x=dates, y=probabilities, name='Predicted Probabilities',
            line=dict(color='red', width=2)
        )
    )

    if file_prefix == 'test' and future_dates is not None and future_probabilities is not None:

        fig_ts_plt.add_trace(
            go.Scatter(
                x=[dates[-1], future_dates[0]], 
                y=[probabilities[-1], future_probabilities[0]],
                name='Future Connection',
                line=dict(color='orange', width=2, dash='dash')
            )
        )
        
        fig_ts_plt.add_trace(
            go.Scatter(
                x=[future_dates[0]], 
                y=[future_probabilities[0]],
                mode='markers',
                marker=dict(color='orange', size=12),
                name='Feb 2025 Prediction'
            )
        )

    fig_ts_plt.update_layout(
        title=f"{title_mapping[file_prefix]}<br>{metrics_str}",
        xaxis_title='Date',
        yaxis_title='Probability [0,1]',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        xaxis=dict(
            dtick="M12",
            tickformat="%Y"
        )
    )
    fig_ts_plt.write_html(f'{plt_dir}/{file_prefix}_timeseries.html')

    predictions_df = pd.DataFrame({
        'date': dates,
        'predicted_probability': probabilities,
        'binary_prediction': binary_predictions,
        'true_value': true_values
    })
    
    if file_prefix == 'test' and future_dates is not None and future_probabilities is not None:
        future_binary_predictions = (future_probabilities >= 0.5).astype(int)
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_probability': future_probabilities,
            'binary_prediction': future_binary_predictions,
            'true_value': [None] * len(future_dates)
        })
        predictions_df = pd.concat([predictions_df, future_df])
    
    predictions_df.to_csv(f'{folder_name}/{file_prefix}_predictions.csv', index=False)

def run_single_model(used_data, t10y2y_lag, n_estimators, max_depth):
    """Runs the recession prediction model for a single parameter configuration."""
    # Define train, validation, and test periods
    train_end = pd.Timestamp('1998-01-01')
    val_end = pd.Timestamp('2004-01-01')
    
    # Create lagged features
    lagged_data = create_lagged_features(used_data, t10y2y_lag)
    lagged_data = lagged_data.dropna()
    
    # Split data
    train_data = lagged_data[lagged_data.index <= train_end]
    val_data = lagged_data[(lagged_data.index > train_end) & (lagged_data.index <= val_end)]
    test_data = lagged_data[lagged_data.index > val_end]
    
    feature_cols = [col for col in lagged_data.columns if col != 'USRECDM']
    X_train, y_train = train_data[feature_cols], train_data['USRECDM']
    X_val, y_val = val_data[feature_cols], val_data['USRECDM']
    X_test, y_test = test_data[feature_cols], test_data['USRECDM']
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    # Generate predictions
    train_pred_proba = rf.predict_proba(X_train_scaled)[:, 1]
    val_pred_proba = rf.predict_proba(X_val_scaled)[:, 1]
    test_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Compute metrics
    metrics_dict = {
        'train_f1': f1_score(y_train, (train_pred_proba >= 0.5).astype(int), zero_division=0),
        'val_f1': f1_score(y_val, (val_pred_proba >= 0.5).astype(int), zero_division=0),
        'test_f1': f1_score(y_test, (test_pred_proba >= 0.5).astype(int), zero_division=0),
        't10y2y_lag': t10y2y_lag,
        'n_estimators': n_estimators,
        'max_depth': max_depth
    }
    
    # Generate future prediction for 2025-02-01
    future_data = create_future_features(used_data, feature_cols, t10y2y_lag)
    if not future_data.empty and not future_data.isna().all().all():
        future_scaled = scaler.transform(future_data)
        future_proba = rf.predict_proba(future_scaled)[:, 1]
        future_dates = pd.date_range(start='2025-02-01', end='2025-02-01', freq='MS')
    else:
        future_dates = None
        future_proba = None
    
    # Define result folder
    result_folder = f'results_lag{t10y2y_lag}_est{n_estimators}_depth{max_depth}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Save results for each set separately
    save_results(
        metrics_dict, result_folder, train_data.index, train_pred_proba, y_train.values,
        rf.feature_importances_, feature_cols, 'train'
    )
    save_results(
        metrics_dict, result_folder, val_data.index, val_pred_proba, y_val.values,
        rf.feature_importances_, feature_cols, 'val'
    )
    save_results(
        metrics_dict, result_folder, test_data.index, test_pred_proba, y_test.values,
        rf.feature_importances_, feature_cols, 'test',
        future_dates=future_dates, future_probabilities=future_proba
    )
    
    print(f"Results saved in: {result_folder}")
    return rf, scaler


# Run a single model's specific results

run_single_model(used_data, t10y2y_lag=20, n_estimators=256, max_depth=8)