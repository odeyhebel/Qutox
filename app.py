"""
PROV MAHAD ULTIMATE AI - REAL ML EDITION
------------------------------------------------------------
Halkan waa AI run ah - MA AHA simulation.
- Xogta: Yahoo Finance (real market candles, real market pairs only)
- Features: RSI, MACD, Bollinger %B, Stochastic, CCI, ADX (dhammaan xisaab dhab ah)
- Model: RandomForestClassifier, la train gareeyay xog taariikhi ah
- Backtest: time-series split (ma isticmaalayo shuffle - lookahead bias looma ogola)
- Natiijada la tuso waa mid DAACAD AH - haddii accuracy-gu hooseeyo, sidaas ayaa la tusayaa.

Sida loo isticmaalo (Pydroid3 / mobile):
  pip install streamlit yfinance pandas numpy scikit-learn --break-system-packages
  streamlit run real_ai_forex_model.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PROV MAHAD - Real AI", layout="wide", initial_sidebar_state="collapsed")

# ──────────────────────────────────────────────────────────────
# 1) INDICATOR CALCULATIONS (real math, no shortcuts)
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
    # %B: 0 = at lower band, 1 = at upper band, 0.5 = at middle
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
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0

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
# 2) DATA FETCH (real market data only - no OTC, no simulation)
# ──────────────────────────────────────────────────────────────

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X",
    "Gold (XAU/USD)": "GC=F",
    "Silver (XAG/USD)": "SI=F",
}

# Yahoo Finance limits how far back you can go per interval.
INTERVAL_PERIOD_MAP = {
    "1m":  "7d",
    "2m":  "60d",
    "5m":  "60d",
    "15m": "60d",
    "1h":  "730d",
    "1d":  "5y",
}

# Yahoo Finance does NOT provide a native 3-minute candle (only 1m, 2m, 5m, 15m...).
# So "3m" is built honestly by resampling real 1-minute candles into 3-minute bars.
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker, interval):
    if interval in RESAMPLE_INTERVALS:
        base_interval, rule = RESAMPLE_INTERVALS[interval]
        base_period = INTERVAL_PERIOD_MAP.get(base_interval, "7d")
        raw = _download_raw(ticker, base_interval, base_period)
        return _resample_ohlc(raw, rule)
    period = INTERVAL_PERIOD_MAP.get(interval, "60d")
    return _download_raw(ticker, interval, period)


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


def build_labels(df, horizon):
    close = df["Close"]
    future_return = close.shift(-horizon) / close - 1
    label = (future_return > 0).astype(int)
    return label, future_return


# ──────────────────────────────────────────────────────────────
# 4) TRAIN + HONEST BACKTEST (time-ordered split, no shuffle)
# ──────────────────────────────────────────────────────────────

def train_and_evaluate(features, labels, test_size=0.25):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score

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
        "n_train": len(train),
        "n_test": len(test),
        "accuracy": accuracy_score(y_test, pred_test),
        "precision": precision_score(y_test, pred_test, zero_division=0),
        "recall": recall_score(y_test, pred_test, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred_test),
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
            rows.append({"threshold": t, "trades_taken": 0, "accuracy": np.nan, "pct_of_data": 0})
            continue
        pred = np.where(buy_mask[mask], 1, 0)
        actual = y_test[mask].values
        acc = (pred == actual).mean()
        rows.append({
            "threshold": t,
            "trades_taken": int(n),
            "accuracy": round(acc * 100, 1),
            "pct_of_data": round(n / len(y_test) * 100, 1),
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 5) STREAMLIT UI
# ──────────────────────────────────────────────────────────────

st.title("🔬 PROV MAHAD ULTIMATE AI — Real ML Edition")
st.caption(
    "AI-kan wuxuu isticmaalaa xog dhab ah oo Yahoo Finance ah, ma isticmaalayo simulation. "
    "Accuracy-ga la tuso waa mid daacad ah — haddii uu hooseeyo, sidaas ayaa la tusayaa."
)

with st.sidebar:
    st.header("⚙️ Settings")
    pair_name = st.selectbox("Trading pair (real market only)", list(PAIRS.keys()))
    interval = st.selectbox("Candle interval", ["1m", "2m", "3m", "5m", "15m", "1h", "1d"], index=4)
    if interval in ("1m", "2m", "3m"):
        st.caption(
            "⚠️ 1m/2m/3m: Yahoo Finance kaliya wuxuu haystaa **7-60 maalmood** oo taariikhi ah "
            "candle-yadan ah - xogtu way ka yar tahay 5m/15m, sidaa darteed model-ku wuu ka "
            "aad u xasaasi (noisy) natiijadu. 3m si dhab ah ayaa looga dhisay 1m candles la isku dara; "
            "2m waa candle dhab ah oo Yahoo ku bixiso."
        )
    horizon = st.slider("Predict N candles ahead", 1, 10, 5)
    test_size = st.slider("Test set size (%)", 10, 40, 25) / 100
    train_btn = st.button("🚀 Train & Backtest (real data)")

if train_btn:
    with st.spinner("Soo qaadaya xog dhab ah oo Yahoo Finance ah..."):
        try:
            raw = fetch_data(PAIRS[pair_name], interval)
        except Exception as e:
            st.error(f"Xogta lama helin: {e}")
            st.stop()

    if raw.empty or len(raw) < 200:
        st.error(
            "Xog kuma filna si model-ka loo tababaro (waxaad u baahan tahay ugu yaraan 200 candle). "
            "Isku day interval kale (tusaale 1h ama 1d)."
        )
        st.stop()

    st.success(f"{len(raw)} candle oo dhab ah ayaa la soo qaaday ({pair_name}, {interval}).")

    feats = build_features(raw)
    labels, future_ret = build_labels(raw, horizon)

    with st.spinner("Model-ka waa la tababarayaa..."):
        model, feat_cols, X_test, y_test, proba_test, metrics = train_and_evaluate(
            feats, labels, test_size
        )

    st.subheader("📊 Natiijada Backtest (daacad ah — xog aan la tababarin)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Precision (BUY)", f"{metrics['precision']*100:.1f}%")
    c3.metric("Recall (BUY)", f"{metrics['recall']*100:.1f}%")
    c4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if not np.isnan(metrics['roc_auc']) else "N/A")

    st.caption(
        f"Train candles: {metrics['n_train']} | Test candles: {metrics['n_test']} "
        "(test set-ku waa xog aan model-ku waligiis arag inta la tababarayay)."
    )

    if metrics["accuracy"] < 0.55:
        st.warning(
            "⚠️ Accuracy-gani wuxuu u dhow yahay 50% (coin-flip). Tani waa caadi — suuqu waa mid "
            "aad noise u badan, oo indicator/ML kaligiis inta badan kuma filna. Ha isticmaalin natiijadan "
            "sida signal la isku halayn karo."
        )
    elif metrics["accuracy"] < 0.60:
        st.info("ℹ️ Accuracy-gani waa xoogaa ka sarreeya coin-flip — waa edge yar, ma aha wax lagu kalsoon karo lacag badan.")
    else:
        st.success("Accuracy-gani wuu wanaagsan yahay marka la barbardhigo caadiga suuqa — weli ha isticmaalin risk management la'aan.")

    st.subheader("🎯 Accuracy sida uu u kala duwan yahay Confidence Threshold")
    st.caption(
        "Xaqiiqda: haddii aad kaliya qaadato trade-yada AI-gu leeyahay confidence sare, "
        "tirada trade-yadu way yaraanaysaa, laakiin accuracy-gu wuu kordhi karaa. Miiskan ayaa ku tusaya taas run ahaan."
    )
    thresh_df = accuracy_by_confidence(y_test.reset_index(drop=True), proba_test, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    st.dataframe(thresh_df, use_container_width=True)

    st.subheader("🧩 Feature Importance (kee indicator ayaa model-ka ugu saameynta badan)")
    importance = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    st.bar_chart(importance)

    st.subheader("🔮 Live Signal (xogta ugu dambeysay)")
    latest_feats = feats.dropna().iloc[[-1]]
    latest_price = raw["Close"].iloc[-1]
    if not latest_feats.empty:
        latest_proba = model.predict_proba(latest_feats[feat_cols])[0, 1]
        sig = "BUY" if latest_proba >= 0.5 else "SELL"
        conf = latest_proba if sig == "BUY" else 1 - latest_proba
        col1, col2 = st.columns(2)
        col1.metric("Signal", sig)
        col2.metric("Model confidence", f"{conf*100:.1f}%")
        st.caption(f"Last close: {latest_price:.5f} | {pair_name} | {interval} candles")
        if conf < 0.6:
            st.warning("Confidence-gani hooseeya — miiska sare ee threshold eeg si aad u ogaato in accuracy-gu ku filan yahay heerkan.")

    st.divider()
    st.caption(
        "⚠️ Disclaimer daacad ah: Natiijadan waa backtest ku salaysan taariikhda. Suuqu wuu isbeddelaa "
        "(market regime change), backtest sare ma dammaanad qaadayo natiijo la mid ah mustaqbalka. "
        "Marna ha isticmaalin lacag aadan awoodin inaad lumiso. Demo account ku tijaabi ugu yaraan "
        "50-100 trade ka hor intaadan lacag dhab ah isticmaalin."
    )
else:
    st.info("Dooro pair iyo interval bidix (sidebar), kadibna riix '🚀 Train & Backtest (real data)'.")
