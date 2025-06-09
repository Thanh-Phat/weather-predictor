import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
df = pd.read_csv("weather_data_2025_5000.csv")

# Làm sạch và tạo nhãn nhị phân
df['weather'] = df['weather'].str.lower()
filtered_df = df[df['weather'].isin(['rain', 'drizzle', 'sun', 'sunny'])].copy()
filtered_df['is_rain'] = filtered_df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)

# Tách đặc trưng và nhãn
X = filtered_df[['temp_max', 'temp_min']]
y = filtered_df['is_rain']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình
model = LogisticRegression()
model.fit(X_train, y_train)

# Hàm dự đoán
def du_doan_thoi_tiet(temp_max, temp_min):
    input_data = pd.DataFrame([[temp_max, temp_min]], columns=['temp_max', 'temp_min'])
    prob = model.predict_proba(input_data)[0][1]  # Xác suất mưa (label=1)
    label = 1 if prob >= 0.5 else 0
    return label, prob

if __name__ == "__main__":
    # Nhập từ người dùng
    temp_max = float(input("🌡️ Nhập nhiệt độ cao nhất (°C): "))
    temp_min = float(input("🌡️ Nhập nhiệt độ thấp nhất (°C): "))

    # Dự đoán
    label, prob = du_doan_thoi_tiet(temp_max, temp_min)
    ket_qua = "RAIN ☔" if label == 1 else "SUN ☀️"

    # In kết quả
    print(f"\n📊 Xác suất có mưa: {prob*100:.2f}%")
    print(f"🌦️  Dự đoán thời tiết: {ket_qua}")
