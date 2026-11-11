import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

st.set_page_config(page_title="Dự đoán giá nhà - K-Means", layout="wide")

# ----------------------------------------
# HÀM TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU
# ----------------------------------------
@st.cache_data
def load_data(train_path, test_path):
    try:
        df_train = pd.read_excel(train_path)
        df_test = pd.read_excel(test_path)
        return df_train, df_test
    except Exception as e:
        st.error(f"Lỗi đọc file: {e}")
        return None, None

# Tìm đường dẫn mặc định
default_train_path = "/content/nhom4_train_data.xlsx"
default_test_path = "/content/nhom4_test_data.xlsx"

# Nếu chạy cục bộ không có thư mục /content thì dùng file tại chỗ
if not os.path.exists(default_train_path):
    default_train_path = "nhom4_train_data.xlsx"
if not os.path.exists(default_test_path):
    default_test_path = "nhom4_test_data.xlsx"

# ----------------------------------------
# GIAO DIỆN SIDEBAR
# ----------------------------------------
st.sidebar.header("⚙️ Cấu hình ứng dụng")

# Cho phép upload file ghi đè
uploaded_train = st.sidebar.file_uploader("Tải lên file Train (Excel)", type=["xlsx"])
uploaded_test = st.sidebar.file_uploader("Tải lên file Test (Excel)", type=["xlsx"])

train_file = uploaded_train if uploaded_train is not None else default_train_path
test_file = uploaded_test if uploaded_test is not None else default_test_path

# Đọc dữ liệu
if os.path.exists(str(train_file)) and os.path.exists(str(test_file)):
    df_train, df_test = load_data(train_file, test_file)
else:
    st.sidebar.warning("Vui lòng đảm bảo các file nhom4_train_data.xlsx và nhom4_test_data.xlsx tồn tại.")
    st.stop()

# Xử lý khuyết khuyết dữ liệu nhẹ
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'HouseAge']
target = 'SalePrice'

df_train = df_train.dropna(subset=features + [target])
df_test = df_test.dropna(subset=features + [target])

# Lựa chọn số cụm K
st.sidebar.subheader("🧠 Tham số K-Means")
# Khởi tạo thuật toán tìm Elbow
X_train_full = df_train[features]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)

# Tính Elbow scores nhanh
inertias = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)

k_val = st.sidebar.slider("Chọn số cụm (K)", min_value=2, max_value=10, value=4)

# Form nhập dữ liệu dự đoán
st.sidebar.subheader("🔮 Nhập thông số nhà cần định giá")
input_OverallQual = st.sidebar.number_input("Chất lượng tổng thể (OverallQual: 1-10)", min_value=1, max_value=10, value=6)
input_GrLivArea = st.sidebar.number_input("Diện tích phòng khách (GrLivArea sqft)", min_value=100, max_value=10000, value=1500)
input_GarageCars = st.sidebar.number_input("Sức chứa Garage (GarageCars)", min_value=0, max_value=5, value=2)
input_TotalBsmtSF = st.sidebar.number_input("Diện tích tầng hầm (TotalBsmtSF sqft)", min_value=0, max_value=10000, value=1000)
input_HouseAge = st.sidebar.number_input("Tuổi thọ của nhà (HouseAge năm)", min_value=0, max_value=150, value=20)

# ----------------------------------------
# HUẤN LUYỆN MÔ HÌNH
# ----------------------------------------
kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
train_clusters = kmeans.fit_predict(X_train_scaled)

# Tính giá trị trung vị SalePrice cho từng cụm
df_train['Cluster'] = train_clusters
cluster_price_map = df_train.groupby('Cluster')[target].median().to_dict()

# Hàm dự đoán
def predict_price(features_input):
    scaled_input = scaler.transform([features_input])
    cluster_assigned = kmeans.predict(scaled_input)[0]
    pred_val = cluster_price_map[cluster_assigned]
    return pred_val, cluster_assigned

# ----------------------------------------
# TRANG CHÍNH (MAIN PANEL) & TABS
# ----------------------------------------
st.title("🏠 Ứng dụng dự đoán giá nhà bằng K-Means")
tab1, tab2, tab3 = st.tabs(["📊 Tổng quan dữ liệu & Phân cụm", "📈 Đánh giá mô hình", "💡 Dự đoán giá nhà"])

with tab1:
    st.header("Tổng quan dữ liệu & Phân cụm")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Xem trước dữ liệu huấn luyện (Train Data)")
        st.dataframe(df_train.head(10))
    with col2:
        st.subheader("Xem trước dữ liệu kiểm thử (Test Data)")
        st.dataframe(df_test.head(10))
    
    st.subheader("Biểu đồ Elbow tối ưu số cụm K")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(K_range, inertias, marker='o', linestyle='--', color='b')
    ax.axvline(x=k_val, color='r', linestyle=':', label=f'K được chọn = {k_val}')
    ax.set_xlabel("Số lượng cụm (K)")
    ax.set_ylabel("Inertia (SSE)")
    ax.set_title("Phương pháp Elbow xác định số cụm")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Đánh giá mô hình trên tập Test")
    
    # Dự đoán trên tập test
    X_test_scaled = scaler.transform(df_test[features])
    test_clusters = kmeans.predict(X_test_scaled)
    
    y_test = df_test[target].values
    y_pred = np.array([cluster_price_map[c] for c in test_clusters])
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Hiển thị chỉ số
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
    m_col2.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")
    m_col3.metric("R² Score (Hệ số xác định)", f"{r2:.4f}")
    
    # Biểu đồ giá trị thực tế vs dự đoán
    st.subheader("Biểu đồ so sánh giá trị Thực tế vs Dự đoán")
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    sns.scatterplot(x=y_test, y=y_pred, hue=test_clusters, palette="viridis", alpha=0.8, ax=ax2)
    
    # Đường chéo lý tưởng
    ideal_line = np.linspace(min(y_test), max(y_test), 100)
    ax2.plot(ideal_line, ideal_line, color='red', linestyle='--', label='Lý tưởng')
    
    ax2.set_xlabel("Giá nhà thực tế ($)")
    ax2.set_ylabel("Giá nhà dự đoán theo cụm ($)")
    ax2.set_title("So sánh giá thực tế vs dự đoán theo cụm (mỗi màu là một cụm phân bổ)")
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    st.header("Dự đoán giá nhà mới")
    input_data = [input_OverallQual, input_GrLivArea, input_GarageCars, input_TotalBsmtSF, input_HouseAge]
    pred_price, assigned_cluster = predict_price(input_data)
    
    st.success(f"### 🏷️ Giá nhà dự đoán đề xuất: **${pred_price:,.2f}**")
    st.info(f"Nhà của bạn được phân loại vào **Cụm số {assigned_cluster}** dựa trên các đặc trưng đã nhập.")
    
    # Thống kê cụm
    st.subheader(f"Thông số trung vị của Cụm {assigned_cluster} so với nhà bạn nhập:")
    cluster_stats = df_train[df_train['Cluster'] == assigned_cluster][features].median()
    
    compare_df = pd.DataFrame({
        'Đặc trưng': features,
        'Giá trị của bạn': input_data,
        f'Trung vị Cụm {assigned_cluster}': cluster_stats.values
    })
    st.table(compare_df)
