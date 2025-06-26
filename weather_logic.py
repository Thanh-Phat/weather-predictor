import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression

# --- Load và chuẩn bị dữ liệu ---
df = pd.read_csv("weather_dataset.csv")
df['weather'] = df['weather'].str.lower()
df['is_rain'] = df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)

# Lọc dữ liệu: giữ lại ngày không mưa hoặc nhiệt độ cao nhất <= 33
filtered_df = df[(df['is_rain'] == 0) | (df['temp_max'] <= 33)].copy()
X = filtered_df[['temp_max', 'temp_min']]
y = filtered_df['is_rain']

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X, y)

# --- Hàm tính xác suất theo nhiệt độ ---
def tinh_xac_suat(temp_max, temp_min):
    # Giả định: nhiệt độ thấp < 25 thì dễ mưa hơn
    base_prob = 1.0 - (temp_min - 20) * 0.05  # giảm 5% mỗi độ tăng
    base_prob = max(0.0, min(1.0, base_prob))  # giới hạn 0-1

    # Nếu trời nóng quá thì giảm thêm
    if temp_max >= 33:
        base_prob *= 0.7

    return base_prob



# --- Hàm dự đoán ---
def du_doan_thoi_tiet(temp_max, temp_min):
    prob = tinh_xac_suat(temp_max, temp_min)
    label = 1 if prob >= 0.5 else 0
    return label, prob

# --- Hàm vẽ biểu đồ ---
def plot_prob_dynamic_bar(temp_max, temp_min_start, temp_min_end):
    if abs(temp_min_start - temp_min_end) > 30:
        if temp_min_start < temp_min_end:
            temp_min_end = temp_min_start + 30
        else:
            temp_min_start = temp_min_end + 30

    temp_min_values = list(range(temp_min_end, temp_min_start - 1, -1))

    probabilities = []

    for tmn in temp_min_values:
        prob = tinh_xac_suat(temp_max, tmn)
        probabilities.append(prob * 100)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(temp_min_values, probabilities, color='gold')
    ax.set_xlabel('Nhiệt độ thấp nhất (°C)')
    ax.set_ylabel('Xác suất mưa (%)')
    ax.set_title(f'Nhiệt độ cao nhất: {temp_max}°C')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return 'data:image/png;base64,' + img_base64
