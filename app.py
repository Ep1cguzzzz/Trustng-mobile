import streamlit as st
from vnstock import stock_historical_data
from datetime import datetime, timedelta

st.title("✅ KẾT NỐI THÀNH CÔNG!")
st.write("Nếu bạn đọc được dòng này nghĩa là Web đã sống lại.")

# Thử lấy dữ liệu nhẹ nhàng
try:
    st.subheader("Test dữ liệu FPT")
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
    
    df = stock_historical_data(symbol='FPT', start_date=start, end_date=end, resolution='1D', type='stock')
    
    if df is not None:
        st.dataframe(df)
        st.success("Dữ liệu đổ về ngon lành! Giờ mới tính tiếp.")
    else:
        st.warning("Không lấy được dữ liệu, nhưng Web vẫn chạy.")
except Exception as e:
    st.error(f"Lỗi lấy dữ liệu: {e}")
