import streamlit as st
import pandas as pd
from vnstock import *
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import requests
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.stats import norm
from datetime import datetime, timedelta

# --- SYSTEM CONFIG (GIAO DIỆN MA TRẬN) ---
st.set_page_config(layout="wide", page_title="PROJECT CHIMERA: THE HYBRID", page_icon="🦁")
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #00FFC8; font-family: 'Verdana', sans-serif; }
    h1, h2, h3 { color: #00FFC8; text-shadow: 0 0 5px #00FFC8; }
    .metric-card { border: 1px solid #333; background: #111; padding: 15px; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 255, 200, 0.1); }
    .stTabs [data-baseweb="tab-list"] { gap: 5px; }
    .stTabs [data-baseweb="tab"] { background-color: #1A1A1A; border: none; color: #888; }
    .stTabs [aria-selected="true"] { background-color: #00FFC8 !important; color: #000 !important; font-weight: bold; }
    .big-score { font-size: 40px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- 1. SIDEBAR CONTROL ---
st.sidebar.title("🦁 CHIMERA CORE")
st.sidebar.info("Hybrid: Titan (SMC/SEPA) + Omega (AI/VaR)")

symbol = st.sidebar.text_input("MÃ CỔ PHIẾU", value="FPT").upper()
years = st.sidebar.slider("Dữ liệu (Năm)", 2, 8, 5)
risk_tol = st.sidebar.select_slider("Khẩu vị rủi ro", ["Thấp", "Trung bình", "Cao"], value="Trung bình")

# --- 2. DATA ENGINE ---
@st.cache_data
def load_data(symbol, years):
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    try:
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution='1D', type='stock')
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df.rename(columns={'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}, inplace=True)
        return df
    except:
        return None

# --- 3. TITAN MODULE (OFFENSE: SEPA & SMC & REGIME) ---
def titan_analysis(df):
    data = df.copy()
    # A. Market Regime (AI Clustering)
    data['Ret'] = data['Close'].pct_change()
    data['Vol'] = data['Ret'].rolling(20).std()
    data.dropna(inplace=True)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(data[['Ret', 'Vol']])
    data['Regime'] = kmeans.labels_
    # Logic label đơn giản
    stats = data.groupby('Regime')['Ret'].mean()
    bull_cluster = stats.idxmax()
    
    regime_status = "BULL" if data['Regime'].iloc[-1] == bull_cluster else "BEAR/SIDEWAYS"

    # B. SEPA Check (Minervini)
    curr = df.iloc[-1]
    sma50 = df['Close'].rolling(50).mean().iloc[-1]
    sma150 = df['Close'].rolling(150).mean().iloc[-1]
    sma200 = df['Close'].rolling(200).mean().iloc[-1]
    high52 = df['Close'].rolling(252).max().iloc[-1]
    
    sepa_pass = (curr['Close'] > sma200) and (sma150 > sma200) and (curr['Close'] > sma50) and (curr['Close'] > high52 * 0.75)
    
    # C. SMC FVG
    fvg_data = df.tail(30).copy()
    last_fvg = None
    for i in range(2, len(fvg_data)):
        if fvg_data['Low'].iloc[i] > fvg_data['High'].iloc[i-2] and fvg_data['Close'].iloc[i-1] > fvg_data['Open'].iloc[i-1]:
            last_fvg = {'Top': fvg_data['Low'].iloc[i], 'Bottom': fvg_data['High'].iloc[i-2], 'Date': fvg_data.index[i]}
            
    return regime_status, sepa_pass, last_fvg

# --- 4. OMEGA MODULE (DEFENSE: ENSEMBLE AI & RISK & ALPHA) ---
def omega_analysis(df):
    data = df.copy()
    
    # A. Alpha Score (Technical Strength)
    data['RSI'] = data.ta.rsi()
    data['EMA20'] = data.ta.ema(length=20)
    score = 0
    if data['RSI'].iloc[-1] > 50: score += 30
    if data['Close'].iloc[-1] > data['EMA20'].iloc[-1]: score += 30
    if data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1]: score += 20
    if data.ta.macd()['MACD_12_26_9'].iloc[-1] > 0: score += 20
    alpha_score = score

    # B. Risk Engine (VaR)
    returns = data['Close'].pct_change().dropna()
    var_95 = norm.ppf(0.05, returns.mean(), returns.std()) # 95% confidence
    
    # C. Ensemble AI Prediction
    data['Log_Ret'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    X = data[['Open', 'High', 'Low', 'Volume', 'RSI', 'Log_Ret']]
    y = data['Close'].shift(-1)
    
    X_train, _, y_train, _ = train_test_split(X[:-1], y[:-1], test_size=0.1, shuffle=False)
    
    # Dùng Voting Regressor (Kết hợp GradientBoosting và RandomForest)
    est1 = GradientBoostingRegressor(n_estimators=50)
    est2 = RandomForestRegressor(n_estimators=50)
    model = VotingRegressor(estimators=[('gb', est1), ('rf', est2)]).fit(X_train, y_train)
    
    pred_price = model.predict(X.iloc[[-1]])[0]
    
    return alpha_score, var_95, pred_price

# --- 5. CHIMERA FUSION CORE (TÍNH ĐIỂM TỔNG HỢP) ---
def calculate_chimera_score(sepa, fvg, regime, alpha, var, curr_price, pred_price):
    # Base score từ Omega (Alpha technicals)
    total_score = alpha 
    
    # 1. Cộng điểm Titan (Offense)
    if sepa: total_score += 20 # Có mẫu hình đẹp
    if regime == "BULL": total_score += 15 # Thị trường ủng hộ
    
    # Check FVG (Vùng mua SMC)
    in_buy_zone = False
    if fvg and (fvg['Bottom'] <= curr_price <= fvg['Top']):
        total_score += 15
        in_buy_zone = True
        
    # 2. Cộng/Trừ điểm Omega (Defense)
    ai_upside = (pred_price - curr_price) / curr_price
    if ai_upside > 0.01: total_score += 10 # AI báo tăng > 1%
    elif ai_upside < -0.01: total_score -= 20 # AI báo giảm -> Trừ nặng
    
    # Phạt nặng nếu rủi ro cao (VaR lỗ sâu hơn -3% 1 ngày)
    if var < -0.03: total_score -= 15
    
    # Cap điểm 0-100
    return max(0, min(100, total_score)), in_buy_zone, ai_upside

# --- 6. DASHBOARD ---
st.title(f"🦁 PROJECT CHIMERA: {symbol}")
df = load_data(symbol, years)

if df is not None:
    # --- RUNNING ALL ENGINES ---
    regime, sepa, fvg = titan_analysis(df)
    alpha, var, ai_pred = omega_analysis(df)
    curr_price = df['Close'].iloc[-1]
    
    chimera_score, in_buy_zone, ai_upside = calculate_chimera_score(sepa, fvg, regime, alpha, var, curr_price, ai_pred)
    
    # --- THE VERDICT (KẾT LUẬN) ---
    action = "QUAN SÁT"
    color = "gray"
    if chimera_score >= 80:
        action = "MUA MẠNH (STRONG BUY)"
        color = "#00FFC8" # Xanh Neon
    elif chimera_score >= 60:
        action = "MUA THĂM DÒ (BUY)"
        color = "#00AAFF" # Xanh Dương
    elif chimera_score <= 40:
        action = "BÁN / ĐỨNG NGOÀI (SELL)"
        color = "#FF4444" # Đỏ
        
    # --- HEADER UI ---
    c1, c2, c3 = st.columns([1.5, 1, 1.5])
    with c1:
        st.caption("QUYẾT NGHỊ CUỐI CÙNG")
        st.markdown(f"<h2 style='color:{color}'>{action}</h2>", unsafe_allow_html=True)
        st.write(f"Giá mục tiêu AI (T+1): **{ai_pred:,.0f}** ({ai_upside*100:.2f}%)")
        
    with c2:
        st.caption("CHIMERA SCORE")
        st.markdown(f"<div class='big-score' style='color:{color}'>{chimera_score:.0f}/100</div>", unsafe_allow_html=True)
        
    with c3:
        st.caption("PHÂN TÍCH RỦI RO (OMEGA SHIELD)")
        risk_pct = var * 100
        st.metric("Max Loss (VaR 95%)", f"{risk_pct:.2f}%", delta="An toàn" if var > -0.03 else "Nguy hiểm", delta_color="normal")
        st.write(f"Thị trường (Titan AI): **{regime}**")

    st.markdown("---")

    # --- VISUALIZATION TABS ---
    tab1, tab2 = st.tabs(["🗺️ BẢN ĐỒ CHIẾN LƯỢC (TITAN VIEW)", "🧮 PHÂN TÍCH ĐỊNH LƯỢNG (OMEGA VIEW)"])
    
    # TAB 1: Dành cho Trader (Chart, SMC, SEPA)
    with tab1:
        fig = go.Figure()
        # Nến
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
        
        # Titan Elements: MA200 (Trend) & FVG
        sma200 = df['Close'].rolling(200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma200, line=dict(color='orange', width=2), name='Trend MA200 (SEPA)'))
        
        if fvg:
            # Vẽ vùng SMC
            fig.add_shape(type="rect", x0=fvg['Date'], x1=df.index[-1], y0=fvg['Bottom'], y1=fvg['Top'], fillcolor="rgba(0, 255, 200, 0.2)", line_width=0)
            fig.add_annotation(x=df.index[-1], y=fvg['Top'], text="SMC BUY ZONE", font=dict(color="#00FFC8"))
            
        fig.update_layout(template="plotly_dark", height=600, title="Biểu đồ Chiến thuật (SMC + SEPA)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Checklist Titan
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.success("✅ SEPA Trend: Tốt") if sepa else col_t1.warning("⚠️ SEPA Trend: Yếu")
        col_t2.success("✅ SMC Zone: Đang ở vùng mua") if in_buy_zone else col_t2.info("ℹ️ SMC Zone: Chờ về vùng gap")
        col_t3.success("✅ Market: Uptrend") if regime=="BULL" else col_t3.error("⛔ Market: Rủi ro")

    # TAB 2: Dành cho Quant (Số liệu, AI, Risk)
    with tab2:
        c_o1, c_o2 = st.columns(2)
        with c_o1:
            st.subheader("🧠 AI Ensemble Forecast")
            st.write("Kết hợp Gradient Boosting & Random Forest để dự báo.")
            delta = ai_pred - curr_price
            st.metric("Dự báo ngày mai", f"{ai_pred:,.0f}", f"{delta:,.0f} ({ai_upside*100:.2f}%)")
            st.progress(int(min(100, max(0, alpha))), text=f"Alpha Score (Sức mạnh kỹ thuật): {alpha}/100")
            
        with c_o2:
            st.subheader("🛡️ Risk Management")
            st.write("Nếu bạn đánh 1 tỷ, với độ tin cậy 95%, ngày mai lỗ tối đa khoảng:")
            loss_amt = abs(var) * 1_000_000_000
            st.error(f"{loss_amt:,.0f} VND")
            st.info("Khuyến nghị: Nếu Chimera Score > 80 hãy vào lệnh, còn không hãy giữ tiền mặt.")

else:
    st.warning("Đang kích hoạt hệ thống Chimera... Vui lòng chờ.")
  
