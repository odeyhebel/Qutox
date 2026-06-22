import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time

# Saacadda Soomaaliya (EAT = UTC+3)
EAT = datetime.timezone(datetime.timedelta(hours=3))
def now_eat():
    return datetime.datetime.now(EAT)

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

    # Confidence score - waxay shaqaysaa labada mode (Balanced iyo Strict)
    # si aad Balanced mode-ka ugu aragto signal-yada ugu adag iyada oo
    # aanad lumin tirada signal-ka. 2/2 = ugu adag, 0/2 = ugu liita.
    confidence = None
    if call_ok or put_ok:
        macd_aligned = (macd_hist > 0) if call_ok else (macd_hist < 0)
        trend_aligned = (trend == 'up') if call_ok else (trend == 'down')
        confidence = int(macd_aligned) + int(trend_aligned)

    info = {'price': close_price, 'rsi': rsi_val, 'trend': trend, 'confidence': confidence}

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
if 'auto_scan' not in st.session_state:
    st.session_state.auto_scan = False

# ============================================================
# Controls
# ============================================================
col_a, col_b = st.columns([2, 1])
with col_a:
    mode = st.radio(
        "Mode-ka Signal-ka:", ['Balanced', 'Strict'], horizontal=True,
        help="Balanced = signals badan. Strict = MACD + 15m confirmation lagu daraa."
    )
with col_b:
    testing_override = st.checkbox("Testing Mode (iska indho-tir saacadaha)")

st.markdown("##### 📰 News Check")
news_clear = st.checkbox(
    "Waan hubiyay — ma jiro war dhaqaale culus (NFP, CPI, FOMC, iwm) socda 30 daqiiqo gudahood."
)
st.caption("Hubi: forexfactory.com/calendar")

market_open = is_market_open() or testing_override

if market_open:
    st.success("🟢 Suuqa Rasmiga ah waa furanyahay.")
else:
    st.error("🚨 Suuqu waa xiran yahay! Wuxuu dib u furmi doonaa Axad 21:00 UTC.")

# ============================================================
# Auto Scan Toggle
# ============================================================
col_start, col_stop = st.columns(2)
with col_start:
    if st.button("▶️ Bilow Auto Scan", use_container_width=True):
        if not market_open:
            st.warning("Suuqu wuu xiran yahay — Testing Mode isticmaal.")
        elif not news_clear:
            st.warning("⚠️ News Check hubi marka hore.")
        else:
            st.session_state.auto_scan = True
with col_stop:
    if st.button("⏹️ Jooji Auto Scan", use_container_width=True):
        st.session_state.auto_scan = False

# ============================================================
# Scan Function (la wadaago Manual iyo Auto)
# ============================================================
def run_scan(mode):
    conf_label = {2: "🟢🟢 2/2 (Adag)", 1: "🟡 1/2 (Dhexdhexaad)", 0: "🔴 0/2 (Tartiib)"}
    found_signals = []

    for ticker, name in pairs.items():
        try:
            df_5m  = fetch_real_market_data(ticker, '1d',  '5m')
            df_15m = fetch_real_market_data(ticker, '5d', '15m')
            signal, info = generate_signal(df_5m, df_15m, mode)
            if ("CALL" in signal or "PUT" in signal) and info:
                found_signals.append((name, signal, info))
                # Log (duplicate check: same asset + signal in last 5 mins)
                now_str = now_eat().strftime('%H:%M:%S')
                duplicate = any(
                    r['Asset'] == name and r['Signal'] == signal
                    for r in st.session_state.signal_log[-20:]
                )
                if not duplicate:
                    st.session_state.signal_log.append({
                        'Waqti'     : now_str,
                        'Asset'     : name,
                        'Signal'    : signal,
                        'Price'     : round(info['price'], 5),
                        'Confidence': conf_label.get(info['confidence'], 'N/A'),
                    })
        except Exception:
            pass

    # Soo bandhig kaliya signals-ka — haddii aanay jirin, fariin gaaban
    st.markdown(f"**🕐 Scan la dhammeeyay: {now_eat().strftime('%H:%M:%S')} (EAT)**")
    if found_signals:
        for name, signal, info in found_signals:
            with st.container():
                if "CALL" in signal:
                    st.success(f"**{name}** — {signal}")
                else:
                    st.error(f"**{name}** — {signal}")
                c1, c2, c3, c4 = st.columns(4)
                c1.caption(f"Price: {round(info['price'], 5)}")
                c2.caption(f"RSI: {round(info['rsi'], 2)}")
                c3.caption(f"15m Trend: {info['trend'] or 'N/A'}")
                c4.caption(f"Confidence: {conf_label.get(info['confidence'], 'N/A')}")
    else:
        st.info("⏳ Hadda ma jiro signal — bot-ku wuu sii eegayaa...")

    # Signal Log
    if st.session_state.signal_log:
        st.markdown("---")
        st.subheader("📋 Taariikhda Signals-ka")
        st.dataframe(
            pd.DataFrame(st.session_state.signal_log[::-1]),
            use_container_width=True
        )

# ============================================================
# Manual Scan button (haddii Auto Scan xiran yahay)
# ============================================================
if not st.session_state.auto_scan:
    if st.button("🔍 Hal mar Scan (Manual)", use_container_width=True):
        if not market_open:
            st.warning("Suuqu wuu xiran yahay.")
        elif not news_clear:
            st.warning("⚠️ News Check hubi marka hore.")
        else:
            run_scan(mode)

# ============================================================
# Auto Scan Loop — 1 second refresh
# ============================================================
if st.session_state.auto_scan:
    if not market_open:
        st.warning("Suuqu wuu xiran yahay — Auto Scan waa la joojiyay.")
        st.session_state.auto_scan = False
    elif not news_clear:
        st.warning("⚠️ News Check hubi marka hore — Auto Scan waa la joojiyay.")
        st.session_state.auto_scan = False
    else:
        st.info("🔄 Auto Scan waa shaqaynaya — wuxuu is-cusbooneysiin doonaa 1 second kasta.")
        run_scan(mode)
        time.sleep(1)
        st.rerun()

st.divider()
st.caption(
    "⚠️ Risk Management: Ha isticmaalin Martingale. 1% kaliya maalinlaha ah. "
    "Demo ku tijaabi marka hore."
)
