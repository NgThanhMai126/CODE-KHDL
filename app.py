import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Hệ thống Phát hiện Gian lận Tài chính",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ỨNG DỤNG DỰ ĐOÁN GIAN LẬN TÀI CHÍNH")
st.markdown(
    "Ứng dụng này sử dụng đại số ma trận (PCA) và học máy không giám sát (Isolation Forest) "
    "để phân tích cấu trúc dữ liệu và phát hiện các giao dịch bất thường."
)

st.sidebar.header("📁 Chọn dữ liệu & cấu hình")
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file CSV dữ liệu giao dịch", type=["csv"]
)

default_paths = [
    Path("financial_anomaly_data.csv"),
    Path.home() / "Downloads" / "financial_anomaly_data.csv",
]

contamination = st.sidebar.slider(
    "Tỷ lệ bất thường dự kiến (contamination)",
    min_value=0.001,
    max_value=0.05,
    value=0.01,
    step=0.001,
)
test_size = st.sidebar.slider(
    "Tỷ lệ dữ liệu Test",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05,
)
use_all_data = st.sidebar.checkbox(
    "Sử dụng toàn bộ dữ liệu (nếu có)",
    value=True,
)
max_samples = st.sidebar.number_input(
    "Số dòng tối đa để lấy mẫu nếu không dùng toàn bộ dữ liệu",
    min_value=1000,
    max_value=1000000,
    value=220000,
    step=5000,
)

@st.cache_data
def load_and_preprocess(source, use_all, max_rows):
    df = pd.read_csv(source, sep=None, engine="python")
    original_rows = len(df)
    if not use_all and original_rows > max_rows:
        df = df.sample(max_rows, random_state=42).reset_index(drop=True)

    required_cols = ['Timestamp', 'Amount', 'TransactionType', 'Location', 'Merchant']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột dữ liệu: {missing_cols}")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    df = df.dropna(subset=required_cols)
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df[['TransactionType', 'Location', 'Merchant']] = df[
        ['TransactionType', 'Location', 'Merchant']
    ].fillna('Unknown')
    df = df.reset_index(drop=True)
    return df, original_rows

source = None
if uploaded_file is not None:
    source = uploaded_file
else:
    for default_path in default_paths:
        if default_path.exists():
            source = default_path
            break

if source is None:
    st.error(
        "Không tìm thấy file dữ liệu. Vui lòng tải lên file CSV hoặc đặt file `financial_anomaly_data.csv` vào thư mục hiện tại hoặc `Downloads`."
    )
    st.stop()

try:
    with st.spinner("⏳ Đang đọc và tiền xử lý dữ liệu..."):
        df_clean, original_rows = load_and_preprocess(source, use_all_data, max_samples)

    st.header("📈 1. Tổng quan Dữ liệu Giao dịch")
    if use_all_data or original_rows <= max_samples:
        st.success(f"Đang phân tích toàn bộ dữ liệu: {original_rows:,} dòng")
    else:
        st.info(
            f"Dữ liệu gốc: {original_rows:,} dòng. Đang lấy mẫu {len(df_clean):,} dòng để phân tích."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng số giao dịch", f"{len(df_clean):,}")
    col2.metric("Giao dịch lớn nhất", f"${df_clean['Amount'].max():,.2f}")
    col3.metric("Số tài khoản duy nhất", df_clean['AccountID'].nunique())

    st.markdown("**Dữ liệu đầu vào**")
    st.dataframe(df_clean.head(8), use_container_width=True)

    cat_cols = ['TransactionType', 'Location', 'Merchant']
    num_cols = ['Amount', 'Hour', 'DayOfWeek']

    df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=True, dtype=float)
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df_clean[num_cols])
    X = np.hstack((X_num_scaled, df_encoded.values))
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_train, X_test, idx_train, idx_test = train_test_split(
        X, np.arange(len(df_clean)), test_size=test_size, random_state=42
    )

    st.header("✂️ 2. Chia dữ liệu Train / Test")
    st.info(
        f"Train: {X_train.shape[0]} dòng | Test: {X_test.shape[0]} dòng"
    )

    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_test_pca = pca.transform(X_test)

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_train)
    test_preds = model.predict(X_test)

    df_test_results = df_clean.iloc[idx_test].copy()
    df_test_results['PCA1'] = X_test_pca[:, 0]
    df_test_results['PCA2'] = X_test_pca[:, 1]
    df_test_results['Dự đoán'] = np.where(
        test_preds == -1,
        'Gian lận (Bất thường)',
        'Bình thường',
    )

    num_anomalies = int((test_preds == -1).sum())
    st.warning(
        f"🚨 Phát hiện {num_anomalies} giao dịch nghi vấn gian lận trên tập Test."
    )

    st.subheader("🔮 3. Trực quan hóa PCA & Phát hiện Gian lận")
    fig_pca = px.scatter(
        df_test_results,
        x='PCA1',
        y='PCA2',
        color='Dự đoán',
        color_discrete_map={
            'Bình thường': '#1E40AF',
            'Gian lận (Bất thường)': '#EF4444',
        },
        hover_data=['AccountID', 'Amount', 'Location', 'TransactionType'],
        opacity=0.75,
        title='Phân tích cấu trúc dữ liệu bằng PCA',
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        fig_box = px.box(
            df_test_results,
            x='Dự đoán',
            y='Amount',
            color='Dự đoán',
            color_discrete_map={
                'Bình thường': '#1E40AF',
                'Gian lận (Bất thường)': '#EF4444',
            },
            title='Phân phối giá trị giao dịch theo dự đoán',
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        df_anomaly = df_test_results[df_test_results['Dự đoán'] == 'Gian lận (Bất thường)']
        if df_anomaly.empty:
            st.info('Không phát hiện ca bất thường nào trên tập Test.')
        else:
            location_counts = (
                df_anomaly['Location'].value_counts().reset_index()
            )
            location_counts.columns = ['Location', 'Số ca gian lận']
            fig_bar = px.bar(
                location_counts,
                x='Location',
                y='Số ca gian lận',
                title='Số ca gian lận theo vị trí',
                color_discrete_sequence=['#DC2626'],
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('📋 4. Chi tiết giao dịch nghi ngờ')
    if not df_anomaly.empty:
        st.dataframe(
            df_anomaly[['Timestamp', 'TransactionID', 'AccountID', 'Amount', 'Merchant', 'TransactionType', 'Location']].head(100),
            use_container_width=True,
        )

except Exception as e:
    st.error(f"💥 Lỗi khi xử lý dữ liệu: {e}")
