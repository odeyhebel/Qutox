import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PROV MAHAD - Ultimate Elite Binary Bot", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────
# 1) LIVE NEWS TRACKER (FOREX FACTORY)
# ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news_calendar():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
        df = pd.DataFrame(data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['date'], utc=True)
        return df
    except Exception:
        return pd.DataFrame()

def check_pair_news(news_df, pair_name):
    if news_df.empty:
        return [], False, 0
    currencies = [c.upper() for c in pair_name.split("/")]
    now = pd.Timestamp.now(tz='UTC')
    country_upper = news_df['country'].astype(str).str.upper()
    impact_upper = news_df['impact'].astype(str).str.upper()
    relevant = news_df[
        (country_upper.isin(currencies)) &
        (impact_upper.isin(['HIGH', 'MEDIUM']))
    ].copy()
    total_relevant = len(relevant)
    if relevant.empty:
        return [], False, 0
    high_risk = False
    upcoming_events = []
    for _, row in relevant.iterrows():
        if pd.isna(row['datetime']):
            continue
        event_time = pd.to_datetime(row['datetime'], utc=True)
        diff_minutes = (event_time - now).total_seconds() / 60.0
        if -30 <= diff_minutes <= 60:
            if str(row['impact']).upper() == 'HIGH':
                high_risk = True
            upcoming_events.append({
                'title': row['title'], 'country': row['country'], 'impact': row['impact'],
                'time': event_time.strftime('%H:%M UTC'), 'diff': int(diff_minutes)
            })
    return upcoming_events, high_risk, total_relevant

# ──────────────────────────────────────────────────────────────
# 2) ADVANCED MOMENTUM & TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line

def calc_bollinger(close, period=20, std_mult=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return ((close - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5)

# ──────────────────────────────────────────────────────────────
# 3) DATA DOWNLOADER & MULTI-TIMEFRAME ENGINE
# ──────────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X", "EUR/JPY": "EURJPY=X", "EUR/CHF": "EURCHF=X",
    "EUR/CAD": "EURCAD=X", "EUR/AUD": "EURAUD=X", "AUD/USD": "AUDUSD=X",
    "AUD/JPY": "AUDJPY=X", "AUD/CHF": "AUDCHF=X", "AUD/CAD": "AUDCAD=X",
    "CAD/CHF": "CADCHF=X", "CAD/JPY": "CADJPY=X", "CHF/JPY": "CHFJPY=X",
    "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X", "USD/JPY": "USDJPY=X",
    "GBP/USD": "GBPUSD=X", "GBP/AUD": "GBPAUD=X"
}

INTERVAL_PERIOD_MAP = {"1m": "7d", "3m": "7d", "5m": "60d", "15m": "60d", "1h": "730d"}
RESAMPLE_INTERVALS = {"3m": ("1m", "3min")}

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
    return df.resample(rule).agg(agg).dropna()

@st.cache_data(ttl=30, show_spinner=False)
def fetch_data(ticker, interval):
    if interval in RESAMPLE_INTERVALS:
        base_interval, rule = RESAMPLE_INTERVALS[interval]
        raw = _download_raw(ticker, base_interval, INTERVAL_PERIOD_MAP.get(base_interval, "7d"))
        df = _resample_ohlc(raw, rule)
    else:
        df = _download_raw(ticker, interval, INTERVAL_PERIOD_MAP.get(interval, "60d"))
    if len(df) > 2:
        df = df.iloc[:-1]
    return df

# ──────────────────────────────────────────────────────────────
# 4) ELITE FEATURE & TARGET ENGINEERING
# ──────────────────────────────────────────────────────────────

def build_features(df):
    out = pd.DataFrame(index=df.index)
    close, high, low, open_p = df["Close"], df["High"], df["Low"], df["Open"]
    out["rsi"] = calc_rsi(close, 14)
    _, _, out["macd_hist"] = calc_macd(close)
    out["bb_percent"] = calc_bollinger(close, 20, 2)
    out["candle_body"] = (close - open_p) / (high - low).replace(0, np.nan)
    out["return_1"] = close.pct_change(1)
    out["return_3"] = close.pct_change(3)
    out["momentum"] = close - close.shift(4)
    return out

def build_labels(df, horizon=1):
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    return (future_return > 0).astype(int), future_return

# ──────────────────────────────────────────────────────────────
# 5) HIGH-PERFORMANCE ENSEMBLE ENGINE (ML + RULES)
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(features, labels, test_size=0.25):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.utils.class_weight import compute_sample_weight

    data = features.copy()
    data["label"] = labels
    data = data.dropna()

    split_idx = int(len(data) * (1 - test_size))
    train, test = data.iloc[:split_idx], data.iloc[split_idx:]

    feat_cols = [c for c in data.columns if c != "label"]
    X_train, y_train = train[feat_cols], train["label"]
    X_test, y_test = test[feat_cols], test["label"]

    model1 = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=1)
    model2 = ExtraTreesClassifier(n_estimators=200, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=1)

    model = VotingClassifier(
        estimators=[('rf', model1), ('et', model2)],
        voting='soft'
    )

    sample_weight = compute_sample_weight("balanced", y_train)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba_test = model.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "buy_rate": float(pred_test.mean()),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_test, proba_test)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return model, feat_cols, X_test, y_test, proba_test, metrics

# ──────────────────────────────────────────────────────────────
# 6) STREAMLIT USER INTERFACE
# ──────────────────────────────────────────────────────────────

st.title("🚀 PROV MAHAD - ULTIMATE BINARY ELITE BOT")
st.caption("Advanced Momentum & Machine Learning Voting Engine for Pocket Option")

news_data = fetch_news_calendar()

with st.sidebar:
    st.header("⚙️ Settings")
    pair_name = st.selectbox("Dooro Pair-ka Lacagta", list(PAIRS.keys()))
    interval = st.selectbox("Dooro Timeframe-ka", ["1m", "3m", "5m", "15m"], index=1)
    predict_horizon = st.slider("Horizon (Candles)", 1, 5, 1)
    test_size_pct = st.slider("Test Size (%)", 10, 50, 25, 5) / 100.0
    st.write("---")
    train_btn = st.button("🔥 RUN ULTIMATE ANALYSIS")

if train_btn:
    upcoming_events, is_high_risk, total_relevant = check_pair_news(news_data, pair_name)
    if is_high_risk:
        st.error(f"🚨 **DIGNIIN NEWS ADAG:** Pair-kan ({pair_name}) wuxuu leeyahay warar High-Impact ah!")
    else:
        st.success(f"✅ **News Check:** Suuqu wuxuu u muuqdaa mid degan marka loo eego wararka.")

    with st.spinner("Falanqeynaya Momentum-ka iyo Xogta Suuqa..."):
        try:
            raw = fetch_data(PAIRS[pair_name], interval)
        except Exception as e:
            st.error(f"Cilad ayaa dhacday: {e}")
            st.stop()

    if raw.empty or len(raw) < 150:
        st.error("Xog ku filan lama helin. Fadlan isku day Timeframe kale.")
        st.stop()

    feats = build_features(raw)
    labels, _ = build_labels(raw, horizon=predict_horizon)
    model, feat_cols, X_test, y_test, proba_test, metrics = train_and_evaluate(feats, labels, test_size=test_size_pct)

    st.subheader("🎯 ULTIMATE LIVE SIGNAL RESULT")
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

        if is_high_risk:
            st.error("⛔ **TALO:** Ka fogow trade-kan sababtoo ah waxaa jira warar halis ah!")
        elif conf < 0.60:
            st.warning(f"⚠️ **Fariin:** Kalsoonidu waa ka hooseysaa 60% ({conf*100:.1f}%). Bot-ku signal ma bixinayo oo waa iska ilaalinayaa buuqa!")
        else:
            st.success(f"🚀 **FURSAD VVIP:** Jihada **{sig}** ayaa la helay iyadoo kalsoonidu tahay ({conf*100:.1f}%)!")

    st.divider()
    st.subheader("📊 Model Performance Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("📉 ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c3.metric("📈 Precision", f"{metrics['precision']*100:.1f}%")
else:
    st.info("Riix badanka '🔥 RUN ULTIMATE ANALYSIS' si aad u bilowdo falanqaynta suuqa.")
