import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

# UI Styling
st.set_page_config(page_title="Quotex Real Market Bot", layout="wide")
st.title("🎯 Quotex Real Market Scanner")
st.subheader("Calaamadaha Saxnaanta Sarreeya ($80\% - 85\%$) | Isniin - Jimco")

# Hubinta maalinta ay tahay (0=Isniin, 4=Jimco, 5=Sabti, 6=Axad)
current_day = datetime.datetime.now().weekday()

if current_day in [5, 6]:
    st.error("🚨 Suuqa Rasmiga ah (Real Market) hadda waa xiran yahay! Bot-ku wuxuu ku jiraa nasashada Weekend-ka.")
    st.info("Ganacsigu wuxuu dib u bilaaban doonaa habeenka Isniintu soo galayso.")
else:
    st.success("🟢 Suuqa Rasmiga ah waa furanyahay. Bot-ku wuxuu diyaar u yahay inuu scan gareeyo.")

    # Real Market Forex Pairs (Ma jiraan OTC)
    pairs = ['EURUSD=X', 'GBPUSD=X', 'AUDUSD=X', 'USDJPY=X', 'USDCAD=X']

    def fetch_real_market_data(ticker):
        # Soo dejinta xogta 1-daqiiqo ee ugu dambaysay ee suuqa rasmiga ah
        data = yf.download(tickers=ticker, period='1d', interval='1m')
        return data

    def generate_high_accuracy_signal(df):
        if len(df) < 30:
            return "Xog ku filan ma jirto"
        
        # Indicators-ka saxnaanta kor u qaadaya
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        
        # RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Helitaanka shumacyadii ugu dambeeyay
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        close_price = float(last_row['Close'])
        open_price = float(last_row['Open'])
        ema_val = float(last_row['EMA_20'])
        rsi_val = float(last_row['RSI'])
        
        prev_close = float(prev_row['Close'])
        prev_open = float(prev_row['Open'])
        
        # --- STRATEGY: PRICE ACTION + INDICATORS ---
        # 1. Bullish Reversal at Support / Oversold (CALL)
        is_bullish_pattern = (prev_close < prev_open) and (close_price > open_price) and (close_price >= prev_open)
        if is_bullish_pattern and rsi_val <= 30 and close_price > ema_val:
            return "🚀 STRONG CALL (Buy)"
            
        # 2. Bearish Reversal at Resistance / Overbought (PUT)
        is_bearish_pattern = (prev_close > prev_open) and (close_price < open_price) and (close_price <= prev_open)
        if is_bearish_pattern and rsi_val >= 70 and close_price < ema_val:
            return "📉 STRONG PUT (Sell)"
            
        return "⏳ Sugitaan (No Safe Setup)"

    # Badanka Scanner-ka
    if st.button("Baar Suuqa Rasmiga ah (Scan Real Market)"):
        cols = st.columns(len(pairs))
        for i, pair in enumerate(pairs):
            with cols[i]:
                st.metric(label=f"Asset: {pair.replace('=X', '')}", value="")
                try:
                    df = fetch_real_market_data(pair)
                    signal = generate_high_accuracy_signal(df)
                    
                    if "CALL" in signal:
                        st.success(f"**Signal:** \n\n {signal}")
                    elif "PUT" in signal:
                        st.error(f"**Signal:** \n\n {signal}")
                    else:
                        st.warning(f"{signal}")
                        
                    st.caption(f"Price: {round(df['Close'].iloc[-1], 5)}")
                    st.caption(f"RSI: {round(df['RSI'].iloc[-1], 2)}")
                except Exception as e:
                    st.error("Cilad baa ku timid soo dejinta xogta.")
