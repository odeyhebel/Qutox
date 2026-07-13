import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PROV MAHAD - Auto ML", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────
# 1) INDICATOR CALCULATIONS
# ──────────────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def calc_bollinger(close, period=20, std_mult=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return percent_b.fillna(0.5)

def calc_stochastic(high, low, close, period=14):
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    k = (close - lowest) / (highest - lowest).replace(0, np.nan) * 100
    return k.fillna(50)

def calc_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mean_dev = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (tp - sma) / (0.015 * mean_dev.replace(0, np.nan))
    return cci.fillna(0)

def calc_adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)

# ──────────────────────────────────────────────────────────────
# 2) DATA FETCH
# ──────────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "EUR/JPY": "EURJPY=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/AUD": "EURAUD=X",
    "AUD/USD": "AUDUSD=X",
    "AUD/JPY": "AUDJPY=X",
    "AUD/CHF": "AUDCHF=X",
    "AUD/CAD": "AUDCAD=X",
    "CAD/CHF": "CADCHF=X",
    "CAD/JPY": "CADJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X",
    "GBP/AUD": "GBPAUD=X"
}

INTERVAL_PERIOD_MAP = {
    "1m":  "7d",
    "2m":  "60d",
    "3m":  "7d",
    "5m":  "60d",
    "15m": "60d",
    "1h":  "730d",
    "1d":  "5y",
}

RESAMPLE_INTERVALS = {
    "3m": ("1m", "3min"),
}

def _download_raw(ticker, interval, period):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def _resample_ohlc(df, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    out = df.resample(rule).agg(agg)
    return out.dropna()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data(ticker, interval):
    if interval in RESAMPLE_INTERVALS:
        base_interval, rule = RESAMPLE_INTERVALS[interval]
        base_period = INTERVAL_PERIOD_MAP.get(base_interval, "7d")
        raw = _download_raw(ticker, base_interval, base_period)
        df = _resample_ohlc(raw, rule)
    else:
        period = INTERVAL_PERIOD_MAP.get(interval, "60d")
        df = _download_raw(ticker, interval, period)
        
    if len(df) > 2:
        df = df.iloc[:-1] # Tuur kandalka aan xirmin
    return df

# ──────────────────────────────────────────────────────────────
# 3) FEATURE + LABEL BUILDING
# ──────────────────────────────────────────────────────────────

def build_features(df):
    out = pd.DataFrame(index=df.index)
    close, high, low = df["Close"], df["High"], df["Low"]

    out["rsi"] = calc_rsi(close, 14)
    macd_line, macd_signal, macd_hist = calc_macd(close)
    out["macd_hist"] = macd_hist
    out["bb_percent"] = calc_bollinger(close, 20, 2)
    out["stoch_k"] = calc_stochastic(high, low, close, 14)
    out["cci"] = calc_cci(high, low, close, 20)
    adx, plus_di, minus_di = calc_adx(high, low, close, 14)
    out["adx"] = adx
    out["di_diff"] = plus_di - minus_di
    out["return_1"] = close.pct_change(1)
    out["return_5"] = close.pct_change(5)
    return out

def build_labels(df, horizon=1):
    close = df["Close"]
    future_return = close.shift(-horizon) / close - 1
    label = (future_return > 0).astype(int)
    return label, future_return

# ──────────────────────────────────────────────────────────────
# 4) TRAIN + BACKTEST
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(features, labels, test_size=0.25):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    data = features.copy()
    data["label"] = labels
    data = data.dropna()

    split_idx = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    feat_cols = [c for c in data.columns if c != "label"]
    X_train, y_train = train[feat_cols], train["label"]
    X_test, y_test = test[feat_cols], test["label"]

    model = RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=25,
        random_state=42, n_jobs=1
    )
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, proba_test)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return model, feat_cols, X_test, y_test, proba_test, metrics

def accuracy_by_confidence(y_test, proba_test, thresholds):
    rows = []
    for t in thresholds:
        buy_mask = proba_test >= t
        sell_mask = proba_test <= (1 - t)
        mask = buy_mask | sell_mask
        n = mask.sum()
        if n == 0:
            continue
        pred = np.where(buy_mask[mask], 1, 0)
        actual = y_test[mask].values
        acc = (pred == actual).mean()
        rows.append({
            "Confidence": f"{int(t*100)}%+",
            "Trades": int(n),
            "Accuracy Natiijo": f"{acc * 100:.1f}%"
        })
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────
# 5) STREAMLIT UI
# ──────────────────────────────────────────────────────────────

st.title("🔬 PROV MAHAD AUTO AI")
st.caption("Koodkani waa Auto-Pilot — Liiska Pairs-kana si buuxda ayaa loo cusbooneysiiyay.")

with st.sidebar:
    st.header("⚙️ Doorashada")
    pair_name = st.selectbox("1. Dooro Lacagta (Pair)", list(PAIRS.keys()))
    interval = st.selectbox("2. Dooro Waqtiga (Timeframe)", ["3m", "5m", "15m", "1h"], index=0)
    
    st.write("---")
    train_btn = st.button("🚀 GET SIGNAL & BACKTEST")

if train_btn:
    with st.spinner("Xogta suuqa ayaa la falanqaynayaa..."):
        try:
            raw = fetch_data(PAIRS[pair_name], interval)
        except Exception as e:
            st.error(f"Cilad ayaa dhacday: {e}")
            st.stop()

    if raw.empty or len(raw) < 200:
        st.error("Xog ku filan oo laga shaqeeyo hadda lama helin. Isku day Timeframe kale.")
        st.stop()

    feats = build_features(raw)
    labels, future_ret = build_labels(raw, horizon=1) 

    model, feat_cols, X_test, y_test, proba_test, metrics = train_and_evaluate(feats, labels, test_size=0.25)

    # 1. LIVE SIGNAL BOX
    st.subheader("🔮 LIVE SIGNAL (Kandalka xiga ee dhalanaya)")
    latest_feats = feats.dropna().iloc[[-1]]
    latest_price = raw["Close"].iloc[-1]
    
    if not latest_feats.empty:
        latest_proba = model.predict_proba(latest_feats[feat_cols])[0, 1]
        sig = "BUY (CALL)" if latest_proba >= 0.5 else "SELL (PUT)"
        conf = latest_proba if sig == "BUY (CALL)" else 1 - latest_proba
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 SIGNAL-KA", sig)
        col2.metric("🎯 CONFIDENCE", f"{conf*100:.1f}%")
        col3.metric("💰 QIIMAHA HADA", f"{latest_price:.5f}")
        
        if conf < 0.70:
            st.warning("⚠️ Digniin: Kalsoonida signal-kan waa ay hoosaysaa (70% ka yar). Ha gelin trade-kan!")
        else:
            st.success("🔥 Signal Adag! Diyaarso Broker-ka markuu kandalku xirmo gal.")

    st.divider()

    # 2. ACCURACY TABLE
    st.subheader("🎯 Jadwalka Saxnaanta (Accuracy Table)")
    thresh_df = accuracy_by_confidence(y_test.reset_index(drop=True), proba_test, [0.5, 0.6, 0.7, 0.8])
    if not thresh_df.empty:
        st.dataframe(thresh_df, use_container_width=True, hide_index=True)
    else:
        st.write("Xog ku filan jadwalka lama hayo hadda.")

else:
    st.info("Dooro Pair iyo Timeframe dhanka bidix ah, ka dibna riix badanka '🚀 GET SIGNAL & BACKTEST'.")
