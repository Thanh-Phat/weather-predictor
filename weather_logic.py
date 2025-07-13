import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dữ liệu
df = pd.read_csv("Data/weatherAUS.csv")
df.columns = df.columns.str.strip()

# Chuyển nhãn về dạng nhị phân
df['RainToday'] = df['RainToday'].map({'No': 0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# # Lọc các cột cần thiết và loại bỏ NaN
# features = ['Humidity3pm', 'Temp3pm']
# df_model = df[features + ['RainTomorrow']].dropna()
df_model = df[['Humidity3pm', 'Temp3pm', 'RainTomorrow']].dropna()
df_model['Hum_Temp'] = df_model['Humidity3pm'] * df_model['Temp3pm']
df_model['Humidity_sq'] = df_model['Humidity3pm'] ** 2
df_model['Temp_sq'] = df_model['Temp3pm'] ** 2

features = ['Humidity3pm', 'Temp3pm', 'Hum_Temp', 'Humidity_sq', 'Temp_sq']

X = df_model[features]
y = df_model['RainTomorrow']

# Chuẩn hóa
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Huấn luyện Logistic Regression
# model = LogisticRegression(class_weight='balanced', max_iter=500)
# model.fit(X_scaled, y)
# Huấn luyện Logistic Regression với trọng số nhẹ
model = LogisticRegression(class_weight={0: 1, 1: 1.5}, max_iter=300)
model.fit(X_train, y_train)

# Hàm tính xác suất mưa
def tinh_xac_suat_ml(humidity, temp):
    hum_temp = humidity * temp
    hum_sq = humidity ** 2
    temp_sq = temp ** 2
    X_input = scaler.transform([[humidity, temp, hum_temp, hum_sq, temp_sq]])
    prob = model.predict_proba(X_input)[0][1]
    return prob

# Hàm dự đoán
def du_doan_thoi_tiet(humidity, temp ):
    if not (0 <= humidity <= 100):
        raise ValueError("Độ ẩm phải nằm trong khoảng 0-100%.")
    prob = tinh_xac_suat_ml(humidity, temp)
    label = 1 if prob >= 0.6 else 0
    return label, prob

# Vẽ biểu đồ xác suất mưa theo độ ẩm
def plot_prob_dynamic_bar(temp, humidity_start, humidity_end):
    if humidity_start > humidity_end:
        humidity_start, humidity_end = humidity_end, humidity_start

    humidity_values = list(range(humidity_start, humidity_end + 1))
    probabilities = []

    for hum in humidity_values:
        prob = tinh_xac_suat_ml(hum, temp)
        probabilities.append(prob * 100)

    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(humidity_values, probabilities, color='skyblue')
    ax.plot(humidity_values, probabilities, color='blue', marker='o', linestyle='-', linewidth=2, label='Xác suất mưa')

    for x, y in zip(humidity_values, probabilities):
        ax.text(x, y + 1, f"{y:.0f}%", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Độ ẩm (%)')
    ax.set_ylabel('Xác suất mưa (%)')
    ax.set_title(f'Nhiệt độ: {temp:.1f}°C')
    ax.set_ylim(0, 100)
    ax.set_xticks(humidity_values)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return 'data:image/png;base64,' + img_base64


# Lưu model và scaler
joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
