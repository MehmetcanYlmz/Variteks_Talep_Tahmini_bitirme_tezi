"""
Variteks Talep Tahmini v2 - Adim 4: Tahmin
==============================================================
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

os.makedirs('results_v2/forecasts', exist_ok=True)

print("=" * 60)
print("ADIM 4 v2: GELECEK 3 AY TAHMINI")
print("=" * 60)

monthly_df = pd.read_csv('results_v2/monthly_sales.csv', index_col=0)
monthly_df.index = pd.PeriodIndex(monthly_df.index, freq='M')
n1_size_df = pd.read_csv('results_v2/n1_size_monthly.csv', index_col=0)
n1_size_df.index = pd.PeriodIndex(n1_size_df.index, freq='M')
features_df = pd.read_csv('results_v2/monthly_sales_features.csv', index_col=0)
features_df.index = pd.PeriodIndex(features_df.index, freq='M')
n1_features_df = pd.read_csv('results_v2/n1_size_features.csv', index_col=0)
n1_features_df.index = pd.PeriodIndex(n1_features_df.index, freq='M')

best_df = pd.read_csv('results_v2/model_comparison/best_models.csv')

all_series = {}
for code in ['X8', 'X2', 'X10', 'X16']: all_series[code] = monthly_df[code]
for col in n1_size_df.columns: all_series[col] = n1_size_df[col]
all_series['N1'] = monthly_df['N1']

all_feat_data = {}
for code in ['X8', 'X2', 'X10', 'X16', 'N1']: all_feat_data[code] = features_df[features_df['series_name'] == code].copy()
for col in n1_size_df.columns: all_feat_data[col] = n1_features_df[n1_features_df['series_name'] == col].copy()

FORECAST_MONTHS = ['2026-05', '2026-06', '2026-07']
FORECAST_LABELS = ['Mayis 2026', 'Haziran 2026', 'Temmuz 2026']
N_FORECAST = 3

def forecast_sarima(series, n_forecast):
    try:
        model = SARIMAX(series.values, order=(1,1,1), seasonal_order=(1,0,1,12), enforce_stationarity=False, enforce_invertibility=False)
        fitted = model.fit(disp=False)
        preds = fitted.forecast(n_forecast)
        ci = fitted.get_forecast(n_forecast).conf_int(alpha=0.05)
        return np.maximum(preds, 0), np.maximum(ci[:,0], 0), ci[:,1]
    except:
        mean, std = series.mean(), series.std()
        return np.full(n_forecast, mean), np.full(n_forecast, max(mean-1.96*std,0)), np.full(n_forecast, mean+1.96*std)

def forecast_hw(series, n_forecast):
    try:
        model = ExponentialSmoothing(series.values, trend='add', seasonal='add', seasonal_periods=12, initialization_method='estimated')
        fitted = model.fit(optimized=True)
        preds = fitted.forecast(n_forecast)
        std_resid = np.std(series.values - fitted.fittedvalues)
        return np.maximum(preds, 0), np.maximum(preds - 1.96*std_resid, 0), preds + 1.96*std_resid
    except:
        mean, std = series.mean(), series.std()
        return np.full(n_forecast, mean), np.full(n_forecast, max(mean-1.96*std,0)), np.full(n_forecast, mean+1.96*std)

def forecast_regression(series, feat_data, n_forecast, model_type='rf'):
    feature_cols = ['month', 'quarter', 'trend', 'month_sin', 'month_cos',
                    'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
                    'rolling_mean_3', 'rolling_mean_6', 'rolling_std_3',
                    'rolling_min_3', 'rolling_max_3', 'expanding_mean']
    train_clean = feat_data.dropna(subset=feature_cols)
    X_train = train_clean[feature_cols].values
    y_train = train_clean['y'].values
    
    if model_type == 'rf': model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    elif model_type == 'lr': model = LinearRegression()
    else: model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    preds = []
    current_values = list(series.values)
    last_period_idx = len(series) - 1
    
    for step in range(n_forecast):
        month = 5 + step
        quarter = (month - 1) // 3 + 1
        trend = last_period_idx + 1 + step
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        lag_1 = current_values[-1]
        lag_2 = current_values[-2]
        lag_3 = current_values[-3]
        lag_6 = current_values[-6] if len(current_values) >= 6 else np.mean(current_values[-3:])
        lag_12 = current_values[-12] if len(current_values) >= 12 else np.mean(current_values)
        rm3 = np.mean(current_values[-3:])
        rm6 = np.mean(current_values[-6:]) if len(current_values) >= 6 else np.mean(current_values[-3:])
        rs3 = np.std(current_values[-3:])
        rmin3 = np.min(current_values[-3:])
        rmax3 = np.max(current_values[-3:])
        exp_mean = np.mean(current_values)
        
        feat_vec = [month, quarter, trend, month_sin, month_cos, lag_1, lag_2, lag_3, lag_6, lag_12, rm3, rm6, rs3, rmin3, rmax3, exp_mean]
        pred = max(model.predict([feat_vec])[0], 0)
        preds.append(pred)
        current_values.append(pred)
        
    std_resid = np.std(y_train - model.predict(X_train))
    preds = np.array(preds)
    return preds, np.maximum(preds - 1.96*std_resid, 0), preds + 1.96*std_resid

def forecast_ma(series, n_forecast, window=3):
    mean, std = series.iloc[-window:].mean(), series.iloc[-window:].std()
    return np.full(n_forecast, mean), np.full(n_forecast, max(mean - 1.96*std, 0)), np.full(n_forecast, mean + 1.96*std)

print("\n--- Tahmin Uretimi (Outliersiz Veri) ---")

MODEL_FUNCTIONS = {
    'MA_3': lambda s, n: forecast_ma(s, n, 3),
    'MA_6': lambda s, n: forecast_ma(s, n, 6),
    'Holt_Winters': forecast_hw,
    'SARIMA': forecast_sarima,
}

forecast_results = []
for series_name, series in all_series.items():
    print(f"\n  [{series_name}]")
    best_model = best_df[best_df['Series'] == series_name].iloc[0]['Best_Model']
    print(f"    Secilen En Iyi Model: {best_model}")
    
    if best_model in MODEL_FUNCTIONS:
        preds, lower, upper = MODEL_FUNCTIONS[best_model](series, N_FORECAST)
    else:
        mt = 'rf' if best_model == 'Random_Forest' else ('lr' if best_model == 'Linear_Regression' else 'xgboost')
        preds, lower, upper = forecast_regression(series, all_feat_data[series_name], N_FORECAST, mt)
        
    # Safeguard against explosive predictions (SARIMA bug)
    max_val = series.max() * 1.5
    if np.any(preds < 0) or np.any(preds > max_val):
        print(f"  [!] {series_name} icin {best_model} patlamasi tespit edildi! Random_Forest'a geciliyor.")
        best_model = 'Random_Forest'
        preds, lower, upper = forecast_regression(series, all_feat_data[series_name], N_FORECAST, 'rf')
        
        # update best_df in memory so the output table shows the fallback model
        best_df.loc[best_df['Series'] == series_name, 'Best_Model'] = 'Random_Forest'
        best_df.to_csv('results_v2/model_comparison/best_models.csv', index=False)
        
    for i in range(N_FORECAST):
        f_val, l_val, u_val = preds[i], lower[i], upper[i]
        
        if np.isnan(l_val): l_val = f_val * 0.4
        if np.isnan(u_val): u_val = f_val * 1.6
        if np.isnan(f_val): f_val = 0
            
        if f_val > 5:
            l_val = max(l_val, f_val * 0.4) # Min %40 kurali
            
        forecast_results.append({
            'Series': series_name,
            'Month': FORECAST_LABELS[i],
            'Period': FORECAST_MONTHS[i],
            'Best_Model': best_model,
            'Forecast': round(f_val),
            'Lower_95': round(l_val),
            'Upper_95': round(u_val),
        })

pd.DataFrame(forecast_results).to_csv('results_v2/forecasts/forecast_table.csv', index=False)
print("\n[OK] Tahminler tamamlandi ve results_v2/forecasts altina kaydedildi.")
