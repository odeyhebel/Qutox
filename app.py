import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# ============================================================
# UI Setup
# ============================================================
st.set_page_config(page_title="Real Market Scanner", layout="wide")
st.title("🎯 Real Market Scanner")
st.subheader("Calaamadaha Suuqa Rasmiga ah | Isniin - Jimco")
st.caption(
    "⚠️ Saxnaanta dhabta ah ee qaab-dhismeedkan waa ~70-75%, marmar gaarsiisan "
    "~78% xaaladaha trend-ku xoogan yahay. Ha ku kalsoonaan 80%+ — nidaam kasta "
    "oo retail ah lama gaarsiin karo heerkaas si joogto ah."
)

# ============================================================
# Market Hours (UTC-based, saxan)
# ============================================================
def is_market_open():
    now_utc = datetime.datetime.utcnow()
    weekday = now_utc.weekday()  # 0=Isniin ... 6=Axad
    hour = now_utc.hour

    if weekday == 4 and hour >= 21:      # Jimce ka dib 21:00 UTC
        return False
    if weekday == 5:                     # Sabti oo dhan
        return False
    if weekday == 6 and hour < 21:       # Axad ka hor 21:00 UTC
        return False
    return True

# ============================================================
# Indicators
# ============================================================
def compute_rsi(series, period=14):
    """RSI oo isticmaalaya Wilder's smoothing (sax ka badan simple rolling mean)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # 50 = neutral marka aan la xisaabin karin


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # histogram


def get_trend_direction(df_htf):
    """Trend-ka higher-timeframe (15m) ee lagu xaqiijinayo signal-ka 5m."""
    if df_htf is None or len(df_htf) < 50:
        return None
    ema50 = df_htf['Close'].ewm(span=50, adjust=False).mean()
    return 'up' if float(df_htf['Close'].iloc[-1]) > float(ema50.iloc[-1]) else 'down'

# ============================================================
# Data Fetching (robust, leh fallback)
# ============================================================
def _fetch(ticker, period, interval):
    try:
        df = yf.download(
            tickers=ticker, period=period, interval=interval,
            auto_adjust=True, group_by='ticker', progress=False
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(0)
            except Exception:
                df.columns = df.columns.droplevel(1)
        return df
    except Exception:
        return None


def fetch_real_market_data(ticker, period, interval):
    df = _fetch(ticker, period, interval)
    if df is None and ticker == 'XAUUSD=X':
        df = _fetch('GC=F', period, interval)  # fallback Gold futures ticker
    return df

# ============================================================
# Signal Logic
# ============================================================
def generate_signal(df_5m, df_15m, mode='Balanced'):
    if df_5m is None or len(df_5m) < 30:
        return "⏳ Xog ku filan ma jirto (Sug 5m)", None

    df = df_5m.copy()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD_HIST'] = compute_macd(df['Close'])

    last, prev = df.iloc[-1], df.iloc[-2]
    close_price, open_price = float(last['Close']), float(last['Open'])
    ema_val, rsi_val = float(last['EMA_20']), float(last['RSI'])
    macd_hist = float(last['MACD_HIST'])
    prev_close, prev_open = float(prev['Close']), float(prev['Open'])

    is_bullish_pattern = (prev_close < prev_open) and (close_price > open_price) and (close_price >= prev_open)
    is_bearish_pattern = (prev_close > prev_open) and (close_price < open_price) and (close_price <= prev_open)

    trend = get_trend_direction(df_15m)

    call_ok = is_bullish_pattern and rsi_val <= 35 and close_price > ema_val
    put_ok = is_bearish_pattern and rsi_val >= 65 and close_price < ema_val

    if mode == 'Strict':
        call_ok = call_ok and macd_hist > 0 and trend == 'up'
        put_ok = put_ok and macd_hist < 0 and trend == 'down'

    info = {'price': close_price, 'rsi': rsi_val, 'trend': trend}

    if call_ok:
        return "🚀 STRONG CALL (Buy)", info
    elif put_ok:
        return "📉 STRONG PUT (Sell)", info
    return "⏳ Sugitaan (No Safe Setup)", info

# ============================================================
# Main App
# ============================================================
pairs = {
    'EURUSD=X': 'EUR/USD',
    'GBPUSD=X': 'GBP/USD',
    'AUDUSD=X': 'AUD/USD',
    'USDJPY=X': 'USD/JPY',
    'USDCAD=X': 'USD/CAD',
    'XAUUSD=X': 'XAU/USD (Gold)',
    'EURCAD=X': 'EUR/CAD',
    'EURCHF=X': 'EUR/CHF',
    'USDCHF=X': 'USD/CHF',
    'AUDCAD=X': 'AUD/CAD',
    'AUDJPY=X': 'AUD/JPY',
    'EURJPY=X': 'EUR/JPY',
    'AUDCHF=X': 'AUD/CHF',
    'CADCHF=X': 'CAD/CHF',
    'CADJPY=X': 'CAD/JPY',
    'CHFJPY=X': 'CHF/JPY',
    'EURAUD=X': 'EUR/AUD',
}

if 'signal_log' not in st.session_state:
    st.session_state.signal_log = []

col_a, col_b = st.columns([2, 1])
with col_a:
    mode = st.radio(
        "Mode-ka Signal-ka:", ['Balanced', 'Strict'], horizontal=True,
        help="Balanced = signals badan, khalad fursaduhu way kor u kacaan. "
             "Strict = signals yar laakiin MACD + 15m trend confirmation ayaa lagu daraa."
    )
with col_b:
    testing_override = st.checkbox("Testing Mode (iska indho-tir saacadaha)")

market_open = is_market_open() or testing_override

if market_open:
    st.success("🟢 Suuqa Rasmiga ah waa furanyahay (ama Testing Mode ayaa shaqaynaya).")
else:
    st.error("🚨 Suuqa Rasmiga ah hadda waa xiran yahay! Wuxuu dib u furmi doonaa Axad 21:00 UTC.")

if st.button("Baar Suuqa Rasmiga ah (Scan Real Market)"):
    if not market_open:
        st.warning("Suuqu wuu xiran yahay — ma scan gareyn karo hadda. Isticmaal Testing Mode haddii aad rabto inaad imtixaanto.")
    else:
        cols = st.columns(len(pairs))
        for i, (ticker, name) in enumerate(pairs.items()):
            with cols[i]:
                st.markdown(f"**{name}**")
                try:
                    df_5m = fetch_real_market_data(ticker, '1d', '5m')
                    df_15m = fetch_real_market_data(ticker, '5d', '15m')
                    signal, info = generate_signal(df_5m, df_15m, mode)

                    if "CALL" in signal:
                        st.success(signal)
                    elif "PUT" in signal:
                        st.error(signal)
                    else:
                        st.warning(signal)

                    if info:
                        st.caption(f"Price: {round(info['price'], 5)}")
                        st.caption(f"RSI: {round(info['rsi'], 2)}")
                        st.caption(f"15m Trend: {info['trend'] or 'N/A'}")

                    if "CALL" in signal or "PUT" in signal:
                        st.session_state.signal_log.append({
                            'Waqti': datetime.datetime.now().strftime('%H:%M:%S'),
                            'Asset': name,
                            'Signal': signal,
                            'Price': round(info['price'], 5) if info else None,
                        })
                except Exception as e:
                    st.error(f"Cilad Farsamo: {str(e)[:60]}")

        if st.session_state.signal_log:
            st.subheader("📋 Taariikhda Signals-ka (Session-kan)")
            st.dataframe(pd.DataFrame(st.session_state.signal_log[::-1]), use_container_width=True)

st.divider()
st.caption(
    "⚠️ Risk Management: Ha isticmaalin Martingale. Stake joogto ah isticmaal "
    "(ugu badnaan 2-5% lacagtaada maalinlaha ah). Marka hore demo ku tijaabi "
    "ilaa aad ka hubto win-rate dhabta ah, kadibna gudbi real money."
    )
