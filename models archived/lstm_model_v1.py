"""Direct LSTM return forecasting for TCS.NS across 1/5/21/30/45 trading days.

Uses a chronological holdout (last ~252 target dates) rather than recursive predictions.
Scaler is fitted only on pre-validation feature rows. Each horizon has its own model.
Research experiment; not a trading recommendation.
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import sys, warnings, json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
TICKER = 'TCS.NS'
HORIZONS = [1, 5, 21, 30, 45]
LOOKBACK = 30
VALIDATION_DAYS = 252
EPOCHS = 30
BATCH_SIZE = 32
SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results' / 'lstm_direct_walkforward_v1'
OUT.mkdir(parents=True, exist_ok=True)

# TensorFlow import is delayed so dependency errors are easy to understand.
def load_tf():
    try:
        import tensorflow as tf
        # Keep routine TensorFlow diagnostics out of the console; actual Python exceptions remain visible.
        tf.get_logger().setLevel('ERROR')
        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)
        except Exception:
            pass
        tf.keras.utils.set_random_seed(SEED)
        return tf
    except Exception as exc:
        raise RuntimeError(
            'TensorFlow could not be imported. Install a TensorFlow build compatible with your Python version. '
            'If no compatible wheel is available for Python 3.14, create a Python 3.12 or 3.13 environment, '
            'then install tensorflow, yfinance, scikit-learn, pandas, numpy, and matplotlib. '
            f'Original error: {exc}'
        ) from exc

def download_data():
    print(f'[1/5] Downloading five years of daily data for {TICKER}...', flush=True)
    raw = yf.download(TICKER, period='5y', interval='1d', auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        raise RuntimeError('No data downloaded. Check internet connection and ticker.')
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise RuntimeError(f'Missing columns from yfinance: {missing}')
    df = raw[needed].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 700:
        raise RuntimeError(f'Only {len(df)} rows downloaded; insufficient for this experiment.')
    print(f'      {len(df)} rows: {df.index.min().date()} to {df.index.max().date()}', flush=True)
    return df

def build_features(df):
    close, volume = df['Close'], df['Volume']
    ret = close.pct_change()
    x = pd.DataFrame(index=df.index)
    for lag in [1,2,3,5,10,15,21,30]:
        x[f'return_lag_{lag}'] = ret.shift(lag-1)
    for w in [2,5,10,21,30]:
        x[f'return_{w}d'] = close.pct_change(w)
    for w in [5,10,21,30,50]:
        x[f'close_to_sma_{w}'] = close / close.rolling(w).mean() - 1
    for w in [5,10,21,30]:
        x[f'volatility_{w}d'] = ret.rolling(w).std()
    x['daily_range_pct'] = (df['High']-df['Low'])/close
    x['open_to_close_pct'] = (df['Close']-df['Open'])/df['Open']
    x['volume_change_1d'] = volume.pct_change()
    x['volume_sma_ratio_5d'] = volume/volume.rolling(5).mean()
    x['volume_sma_ratio_21d'] = volume/volume.rolling(21).mean()
    x['day_of_week'] = df.index.dayofweek
    x['month'] = df.index.month
    return x.replace([np.inf,-np.inf], np.nan)

def make_sequences(features, target_ret, target_price, target_dates, scaler, lookback):
    Xs, ys, prices, origins, dates = [], [], [], [], []
    arr = scaler.transform(features)
    for i in range(lookback-1, len(features)):
        if not (np.isfinite(target_ret.iloc[i]) and np.isfinite(target_price.iloc[i]) and pd.notna(target_dates.iloc[i])):
            continue
        seq = arr[i-lookback+1:i+1]
        if np.isfinite(seq).all():
            Xs.append(seq)
            ys.append(float(target_ret.iloc[i]))
            prices.append(float(target_price.iloc[i]))
            origins.append(features.index[i])
            dates.append(pd.Timestamp(target_dates.iloc[i]))
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys), np.asarray(prices), np.asarray(origins), np.asarray(dates)

def build_model(tf, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, n_features)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

def main():
    tf = load_tf()
    df = download_data()
    feat = build_features(df)
    close = df['Close']
    last_date = df.index[-1]
    validation_start = df.index[-VALIDATION_DAYS]
    metric_rows = []
    print('[2/5] Building chronological holdout...', flush=True)
    print(f'      Validation target-date window starts around {validation_start.date()}', flush=True)
    for h in HORIZONS:
        print(f'\n[3/5] Training direct LSTM for {h}-day horizon...', flush=True)
        target_ret = close.shift(-h)/close - 1
        target_price = close.shift(-h)
        target_date = pd.Series(df.index, index=df.index).shift(-h)
        valid_feature_rows = feat.notna().all(axis=1)
        train_rows = valid_feature_rows & target_date.notna() & (target_date < validation_start) & target_ret.notna()
        test_rows = valid_feature_rows & target_date.notna() & (target_date >= validation_start) & (target_date <= last_date) & target_ret.notna()
        if train_rows.sum() < 300 or test_rows.sum() < 50:
            raise RuntimeError(f'Insufficient train/test rows for {h}D: train={train_rows.sum()}, test={test_rows.sum()}')
        # Fit scaler on historical rows only; no validation-period statistics enter preprocessing.
        scaler = StandardScaler()
        scaler.fit(feat.loc[train_rows])
        X_all, y_all, p_all, origins_all, dates_all = make_sequences(feat, target_ret, target_price, target_date, scaler, LOOKBACK)
        train_mask = np.array([pd.Timestamp(d) < validation_start for d in dates_all])
        test_mask = np.array([(pd.Timestamp(d) >= validation_start and pd.Timestamp(d) <= last_date) for d in dates_all])
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        actual_prices = p_all[test_mask]
        origins = origins_all[test_mask]
        dates = dates_all[test_mask]
        if len(X_train) < 250 or len(X_test) < 50:
            raise RuntimeError(f'Insufficient sequence rows for {h}D: train={len(X_train)}, test={len(X_test)}')
        # Chronological tail of training data is used for early stopping; no random split.
        cut = max(int(len(X_train)*0.9), 1)
        X_fit, y_fit = X_train[:cut], y_train[:cut]
        X_es, y_es = X_train[cut:], y_train[cut:]
        model = build_model(tf, X_train.shape[2])
        callbacks = []
        if len(X_es) >= 20:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True))
            hist = model.fit(X_fit, y_fit, validation_data=(X_es,y_es), epochs=EPOCHS, batch_size=BATCH_SIZE,
                             shuffle=False, verbose=0, callbacks=callbacks)
            epochs_run = len(hist.history['loss'])
        else:
            hist = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False, verbose=0)
            epochs_run = len(hist.history['loss'])
        # Call the model directly instead of model.predict(). Each horizon creates a new model;
        # direct inference avoids repeatedly constructing Keras predict functions and reduces
        # tf.function retracing overhead across horizons.
        pred_ret = model(X_test, training=False).numpy().reshape(-1)
        origin_prices = np.array([float(close.loc[pd.Timestamp(d)]) for d in origins])
        pred_prices = origin_prices * (1 + pred_ret)
        naive_prices = origin_prices
        mom_returns = np.array([float(close.loc[pd.Timestamp(o)]/close.shift(h).loc[pd.Timestamp(o)] - 1) for o in origins])
        mom_prices = origin_prices * (1 + mom_returns)
        actual_returns = y_test
        pred_pct, actual_pct, mom_pct = pred_ret*100, actual_returns*100, mom_returns*100
        mae = lambda a,b: float(mean_absolute_error(a,b))
        rmse = lambda a,b: float(np.sqrt(mean_squared_error(a,b)))
        direction = lambda a,b: float((np.sign(a)==np.sign(b)).mean()*100)
        x_mae, n_mae, m_mae = mae(actual_prices,pred_prices), mae(actual_prices,naive_prices), mae(actual_prices,mom_prices)
        x_rmae, n_rmae, m_rmae = mae(actual_pct,pred_pct), mae(actual_pct,np.zeros_like(actual_pct)), mae(actual_pct,mom_pct)
        pred_df = pd.DataFrame({
            'forecast_origin_date':[pd.Timestamp(d).date().isoformat() for d in origins],
            'target_date':[pd.Timestamp(d).date().isoformat() for d in dates],
            'horizon_trading_days':h, 'origin_close':origin_prices,
            'actual_close':actual_prices, 'lstm_predicted_return_pct':pred_pct,
            'lstm_predicted_close':pred_prices, 'naive_predicted_close':naive_prices,
            'momentum_trailing_return_pct':mom_pct, 'momentum_predicted_close':mom_prices,
            'actual_return_pct':actual_pct
        })
        pred_df.to_csv(OUT/f'predictions_{h}d.csv',index=False)
        metric_rows.append({
            'ticker':TICKER,'forecast_method':'direct_lstm','horizon_trading_days':h,
            'validation_predictions':len(X_test),'validation_target_start':pred_df.target_date.min(),
            'validation_target_end':pred_df.target_date.max(),'epochs_run':epochs_run,
            'lstm_price_mae':x_mae,'naive_price_mae':n_mae,'momentum_price_mae':m_mae,
            'lstm_price_rmse':rmse(actual_prices,pred_prices),'naive_price_rmse':rmse(actual_prices,naive_prices),
            'momentum_price_rmse':rmse(actual_prices,mom_prices),
            'lstm_return_mae_percentage_points':x_rmae,'naive_return_mae_percentage_points':n_rmae,
            'momentum_return_mae_percentage_points':m_rmae,
            'lstm_directional_accuracy_pct':direction(pred_pct,actual_pct),
            'momentum_directional_accuracy_pct':direction(mom_pct,actual_pct),
            'lstm_price_mae_improvement_vs_naive_pct':(n_mae-x_mae)/n_mae*100,
            'lstm_price_mae_improvement_vs_momentum_pct':(m_mae-x_mae)/m_mae*100
        })
        model.save(OUT/f'lstm_{h}d.keras')
        with open(OUT/f'lstm_{h}d_metadata.json','w',encoding='utf-8') as f:
            json.dump({'ticker':TICKER,'horizon_trading_days':h,'lookback_days':LOOKBACK,
                       'feature_columns':list(feat.columns),'validation_start_approx':validation_start.date().isoformat(),
                       'protocol':'fixed chronological holdout; train target dates before validation start; scaler fit on pre-validation rows only',
                       'warning':'Research estimate only; not a trading recommendation.'},f,indent=2)
        plot = pred_df.sort_values('target_date')
        plt.figure(figsize=(11,5))
        plt.plot(pd.to_datetime(plot.target_date),plot.actual_close,label='Actual close')
        plt.plot(pd.to_datetime(plot.target_date),plot.lstm_predicted_close,label='LSTM')
        plt.plot(pd.to_datetime(plot.target_date),plot.naive_predicted_close,label='Naive baseline',alpha=.8)
        plt.plot(pd.to_datetime(plot.target_date),plot.momentum_predicted_close,label='Momentum benchmark',alpha=.8)
        plt.title(f'{TICKER}: {h}-Trading-Day Direct LSTM vs Benchmarks')
        plt.xlabel('Target date'); plt.ylabel('Adjusted closing price (INR)')
        plt.legend(); plt.grid(True,alpha=.25); plt.tight_layout()
        plt.savefig(OUT/f'forecast_{h}d.png',dpi=150); plt.close()
        print(f'      Predictions: {len(X_test)} | epochs: {epochs_run}',flush=True)
        print(f'      Price MAE: LSTM ₹{x_mae:.2f} | naive ₹{n_mae:.2f} | momentum ₹{m_mae:.2f}',flush=True)
    pd.DataFrame(metric_rows).to_csv(OUT/'metrics.csv',index=False)
    print('\n[5/5] COMPLETE',flush=True)
    print(f'Outputs saved to: {OUT}',flush=True)
    print(pd.DataFrame(metric_rows)[['horizon_trading_days','lstm_price_mae','naive_price_mae','momentum_price_mae','lstm_directional_accuracy_pct']].to_string(index=False),flush=True)
    print('Caution: holdout protocol differs from XGBoost walk-forward protocol; compare cautiously until validation is harmonized.',flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nERROR: {exc}',file=sys.stderr,flush=True)
        raise
