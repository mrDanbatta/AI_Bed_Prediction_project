import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

FORECAST_DAYS = 14
TEST_DAYS = 28
HISTORY_DAYS = 90


def process_single_ward(df, hospital, ward_name, TEST_DAYS=28, FORECAST_DAYS=14):

    # preparing the data by resampling and filling missing values
    ts = df.copy()
    ts = ts.resample('D').mean().ffill().bfill()

    # splitting the data into training and testing sets
    train_cutoff = ts.index.max() - pd.Timedelta(days=TEST_DAYS)
    train = ts[ts.index <= train_cutoff]
    test = ts[ts.index > train_cutoff]
    

    if os.path.exists(f'models/{hospital}_{ward_name}_sarima_model.pkl'):
        # load pre-trained model
        model_results = joblib.load(f'models/{hospital}_{ward_name}_sarima_model.pkl')
        # evaluate model
        results_dict = evaluate_sarima_model(train, test, model_results)

        forecast_df = generate_forecast(model_results, ts, ward_name, hospital, FORECAST_DAYS)
    else:
        # hyperparameter tuning to find the best SARIMA parameters
        best_params, best_aic, best_model_results = find_best_sarima_params(train)
        
        # fit SARIMA model
        model_results = fit_model(best_params, (1, 1, 1, 7), train)
        
        # evaluate model
        results_dict = evaluate_sarima_model(train, test, model_results)

        forecast_df = generate_forecast(model_results, ts, ward_name, hospital, FORECAST_DAYS)

        # save the trained model
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_results, f'models/{hospital}_{ward_name}_sarima_model.pkl')
    
    return forecast_df, results_dict, train, test



def generate_forecast(sarima_model, ts, ward_name, hospital, forecast_days=14):
    """
    Generate SARIMA forecast for a given ward and hospital.
    
    Parameters:
    -----------
    sarima_model : SARIMAX results object
        Fitted SARIMA model
    ts : pd.Series
        Complete time series data
    ward_name : str
        Name of the ward
    hospital : str
        Hospital identifier
    forecast_days : int
        Number of days to forecast (default: 14)
    
    Returns:
    --------
    pd.DataFrame : Forecast dataframe with predictions and confidence intervals
    """
    
    # Generate forecast
    forecast = sarima_model.get_forecast(steps=forecast_days)
    forecast_df = forecast.summary_frame()
    
    # Create future dates for the forecast
    forecast_dates = pd.date_range(
        start=ts.index.max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq='D'
    )
    
    forecast_df.index = forecast_dates
    
    # Add metadata
    forecast_df['ward'] = ward_name
    forecast_df['hospital'] = hospital
    forecast_df['date'] = forecast_df.index
    
    # Rename columns for clarity
    forecast_df = forecast_df.rename(
        columns={
            'mean': 'predicted_occupied_beds',
            'mean_ci_lower': 'ci_lower',
            'mean_ci_upper': 'ci_upper'
        }
    )

   
    
    # print(f"Forecast for the next {forecast_days} days:")
    # print(f"Forecast period from {forecast_df.index.min().date()} to {forecast_df.index.max().date()}")
    # print(f"Forecast results:")
    # print(forecast_df[['date', 'predicted_occupied_beds', 'ci_lower', 'ci_upper']].round(1))
    
    return forecast_df


def find_best_sarima_params(train, p_range=range(0, 3), d_range=range(0, 3), q_range=range(0, 3), seasonal_order=(1, 1, 1, 7)):
    """
    Find the best SARIMA parameters using grid search based on AIC.
    
    Parameters:
    -----------
    train : pd.Series
        Training time series data
    p_range : range
        Range of p values to test (default: 0-2)
    d_range : range
        Range of d values to test (default: 0-2)
    q_range : range
        Range of q values to test (default: 0-2)
    seasonal_order : tuple
        SARIMA seasonal order (P, D, Q, s) (default: (1, 1, 1, 7))
    
    Returns:
    --------
    tuple : (best_params, best_aic, best_model_results)
        - best_params: tuple of (p, d, q) with lowest AIC
        - best_aic: AIC value of best model
        - best_model_results: fitted SARIMAX results object
    """
    pdq = list(itertools.product(p_range, d_range, q_range))
    
    best_aic = np.inf
    best_params = None
    best_model_results = None
    
    for param in pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(
                train,
                order=param,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            results = model.fit(disp=False, maxiter=1000)
            # print(f"SARIMA{param} - AIC:{results.aic:.2f} BIC:{results.bic:.2f} HQIC:{results.hqic:.2f}")
            
            # Track the best model based on AIC
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_model_results = results
        except:
            continue
    
    # print(f"\nBest SARIMA Parameters: {best_params}")
    # print(f"Best AIC: {best_aic:.2f}")
    
    return best_params, best_aic, best_model_results


def fit_model(order, seasonal_order, train_data):
    """
    Fit SARIMA model with given order and seasonal order.
    
    Parameters:
    -----------
    order : tuple
        SARIMA order (p, d, q)
    seasonal_order : tuple
        SARIMA seasonal order (P, D, Q, s)
    train_data : pd.Series
        Training time series data
    
    Returns:
    --------
    SARIMAX results object : Fitted SARIMA model results
    """
    model = sm.tsa.statespace.SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False, maxiter=1000)
    return results



def evaluate_sarima_model(train, test, best_model_results):
    """
    Generate predictions and evaluate SARIMA model performance.
    
    Parameters:
    -----------
    train : pd.Series
        Training time series data
    test : pd.Series
        Test time series data
    best_model_results : SARIMAX results object
        Fitted SARIMA model results
    
    Returns:
    --------
    dict : Dictionary containing predictions, metrics, and forecast object
    """
    # generate predictions on test set
    test_forecast = best_model_results.get_forecast(steps=len(test))
    test_predictions = test_forecast.predicted_mean

    # calculate metrics
    mae = mean_absolute_error(test, test_predictions)
    mape = np.mean(np.abs((test - test_predictions) / test)) * 100

    # naive baseline persistence model
    naive_forecast = pd.Series([train.iloc[-1]] * len(test), index=test.index)
    naive_mape = np.mean(np.abs((test - naive_forecast) / test)) * 100

    # print(f"SARIMA Model MAE: {mae:.2f} beds")
    # print(f"SARIMA Model MAPE: {mape:.2f}%")
    # print(f"Naive Baseline MAPE: {naive_mape:.2f}%")
    # print(f"Improvement over Naive Baseline: {naive_mape - mape:.2f}%")
    
    return {
        'test_forecast': test_forecast,
        'test_predictions': test_predictions,
        'mae': mae,
        'mape': mape,
        'naive_mape': naive_mape
    }

def plot_forecast(ts, train, test, forecast_df, results_dict, hospital, ward_name):
    # prepare data for plotting
    last_historical_date = ts.index.max()
    last_historical_value = ts.iloc[-1]
    test_predictions = results_dict['test_predictions']
    train_cutoff = train.index.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

    # plot 1: Historical, Predictions, and Forecast
    ax1.plot(ts.index[- HISTORY_DAYS:], ts.values[-HISTORY_DAYS:], label='Historical Occupied Beds', linewidth=2, color='blue')
    ax1.plot(test.index, test.values, label = 'Actuals', linewidth=2, color='orange')
    ax1.plot(test.index, test_predictions.values, label='SARIMA Predictions', linewidth=2, color='green')

    forecast_dates_with_last = pd.DatetimeIndex([last_historical_date]).union(forecast_df.index)
    forecast_values_with_last = [last_historical_value] + list(forecast_df["predicted_occupied_beds"])
    ax1.plot(forecast_dates_with_last, forecast_values_with_last, label=f'{FORECAST_DAYS}-day Forecast', linewidth=2, color='red')

    ax1.fill_between(forecast_df.index,
                    forecast_df['ci_lower'],
                    forecast_df['ci_upper'],
                    alpha=0.2, label='Forecast Confidence Interval')

    ax1.axvline(x = train_cutoff, color = 'purple', linestyle = '--', alpha=0.7, label='Train/Test Split')

    ax1.set_title(f'SARIMA Model Predictions and {FORECAST_DAYS}-day Forecast - {ward_name} at {hospital}', fontsize=14, fontweight='bold')

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Occupied Beds', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # add metric box
    metrics_text = f"Test MAPE: {results_dict['mape']:.2f}%\nNaive MAPE: {results_dict['naive_mape']:.2f}%\nImprovement: {results_dict['naive_mape'] - results_dict['mape']:.2f}%"
    ax1.text(0.02, 0.95, metrics_text, 
            transform=ax1.transAxes, 
            va = 'top',
            bbox = dict(boxstyle='round', facecolor='white', alpha=0.5),
            fontsize=10,)

    # format x-axis dates
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.tick_params(axis='x', rotation=45)

    # plot 2: Forecast only
    zoom_start = forecast_df.index.min() - pd.Timedelta(days=7)
    ax2.set_xlim([zoom_start, forecast_df.index.max() + pd.Timedelta(days=1)])

    # get data from zoomed period
    zoom_mask = ts.index >= zoom_start
    zoom_historical = ts[zoom_mask]

    ax2.plot(zoom_historical.index, zoom_historical.values, 
            label='Historical Occupied Beds', linewidth=2, color='blue')

    ax2.plot(forecast_dates_with_last, forecast_values_with_last, '--',
                label=f'{FORECAST_DAYS}-day Forecast', linewidth=2, color='red')

    ax2.fill_between(forecast_df.index,
                    forecast_df['ci_lower'],
                    forecast_df['ci_upper'],
                    alpha=0.2, label='Forecast Confidence Interval', color='red')

    # add today's line
    ax2.axvline(x=last_historical_date, color = 'green', linestyle='--', alpha=0.7, label="Today's Date")

    # formatting
    ax2.set_title(f'{FORECAST_DAYS}-day Forecast Zoomed View - {ward_name} at {hospital}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Occupied Beds', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # format x-axis dates
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # add forecast values as text
    for i, date in enumerate(forecast_df.index): #show first 5 forecast values
        value = forecast_df.loc[date, "predicted_occupied_beds"]
        ax2.text(date, value + 0.5, f"{value:.1f}", ha='center', va='bottom', fontsize=9, color='red')

    plt.tight_layout()
    return fig