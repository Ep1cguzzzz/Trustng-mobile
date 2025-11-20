import streamlit as st
import pandas as pd
from vnstock import *
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

# --- CẤU HÌNH TITAN SPEED ---
st.set_page_config(layout="wide", page_title="TITAN SPEED: QUANT LITE", page_icon="⚡")
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #00FF00; font-family: sans-serif; }
    .metric-card { background: #111; border: 1px solid #333; padding: 10px; border-radius: 5px; }
    h1, h2 { color: #00FF00; }
</style>
""", unsafe_allow_html=True)

# --- 1. SIDEBAR ---
st.sidebar.title("⚡ TITAN SPEED")
st.sidebar.info("High Performance Quant System")
symbol = st.sidebar.text_input("MÃ CỔ PHIẾU", value="FPT").upper()
days = st.sidebar.slider("Số ngày dữ liệu", 100, 1000, 365)

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data(symbol, days):
    try:
        end = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        return df
    except:
        return None

# --- 3. QUANT LOGIC (FAST) ---
def analyze_stock(df):
    data = df.copy()
    # Indicator
    data['RSI'] = data.ta.rsi()
    data['EMA20'] = data.ta.ema(length=20)
    data['EMA50'] = data.ta.ema(length=50)
    data['EMA200'] = data.ta.ema(length=200)
    
    # A. SEPA TREND CHECK
    curr = data.iloc[-1]
    sepa_ok = (curr['Close'] > curr['EMA200']) and (curr['EMA50'] > curr['EMA200'])
    
    # B. SMC FVG (Gap)
    last_fvg = None
    # Lấy 20 nến gần nhất soi cho nhanh
    sub = data.tail(20)
    for i in range(2, len(sub)):
        if sub['Low'].iloc[i] > sub['High'].iloc[i-2] and sub['Close'].iloc[i-1] > sub['Open'].iloc[i-1]:
            last_fvg = {'Top': sub['Low'].iloc[i], 'Bottom': sub['High'].iloc[i-2], 'Date': sub.index[i]}
    
    # C. TITAN SCORE (Thay AI bằng Logic thuật toán)
    score = 50
    if sepa_ok: score += 20
    if curr['RSI'] > 50: score += 10
    if curr['Close'] > curr['EMA20']: score += 10
    if curr['Volume'] > data['Volume'].rolling(20).mean().iloc[-1]: score += 10
    
    return data, score, sepa_ok, last_fvg

# --- 4. GIAO DIỆN ---
st.title(f"⚡ TITAN SPEED: {symbol}")
df = load_data(symbol, days)

if df is not None:
    data, score, sepa, fvg = analyze_stock(df)
    curr_price = data['Close'].iloc[-1]
    
    # HUD
    c1, c2, c3 = st.columns(3)
    c1.metric("GIÁ HIỆN TẠI", f"{curr_price:,.0f}")
    c2.metric("TITAN SCORE", f"{score}/100", "MUA MẠNH" if score > 80 else "QUAN SÁT")
    c3.metric("XU HƯỚNG (SEPA)", "TĂNG TRƯỞNG" if sepa else "YẾU")
    
    # CHART
    st.subheader("Biểu đồ phân tích")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA200'], line=dict(color='orange'), name='MA200 (Trend)'))
    
    if fvg:
        fig.add_shape(type="rect", x0=fvg['Date'], x1=data.index[-1], y0=fvg['Bottom'], y1=fvg['Top'], fillcolor="green", opacity=0.3, line_width=0)
        fig.add_annotation(x=data.index[-1], y=fvg['Top'], text="SMC BUY ZONE", showarrow=False, font=dict(color="green"))
        
    fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
else:
    st.warning("Đang tải dữ liệu siêu tốc...")
