from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- Load và chuẩn bị dữ liệu ---
df = pd.read_csv("weather_data_2025_5000.csv")
df['weather'] = df['weather'].str.lower()
filtered_df = df[df['weather'].isin(['rain', 'drizzle', 'sun', 'sunny'])].copy()
filtered_df['is_rain'] = filtered_df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)

X = filtered_df[['temp_max', 'temp_min']]
y = filtered_df['is_rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# --- Hàm dự đoán ---
def du_doan_thoi_tiet(temp_max, temp_min):
    input_data = pd.DataFrame([[temp_max, temp_min]], columns=['temp_max', 'temp_min'])
    prob = model.predict_proba(input_data)[0][1]  # xác suất mưa
    label = 1 if prob >= 0.5 else 0
    return label, prob

# --- Hàm vẽ biểu đồ trả về base64 ---
def plot_prob_dynamic_bar(temp_max, temp_min_start, temp_min_end):
    
    # Đảm bảo thứ tự hợp lệ
    if abs(temp_min_start - temp_min_end) > 30:
        if temp_min_start < temp_min_end:
            temp_min_end = temp_min_start + 30
        else:
            temp_min_start = temp_min_end + 30


    temp_min_values = list(range(temp_min_start, temp_min_end + 1))
    probabilities = []

    for tmn in temp_min_values:
        input_data = pd.DataFrame([[temp_max, tmn]], columns=['temp_max', 'temp_min'])
        prob = model.predict_proba(input_data)[0][1] * 100
        probabilities.append(prob)

    # Đóng tất cả các biểu đồ cũ trước khi tạo mới
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
    plt.close(fig)  # Đóng riêng figure vừa tạo
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return 'data:image/png;base64,' + img_base64


# --- Route chính ---
@app.route('/', methods=['GET', 'POST'])
def home():
    temp_max = temp_min = prob = ket_qua = None
    try:
        if request.method == 'POST':
            temp_max = float(request.form['temp_max'])
            temp_min = float(request.form['temp_min'])

            label, prob = du_doan_thoi_tiet(temp_max, temp_min)
            ket_qua = "RAIN ☔" if label == 1 else "SUN ☀️"

            temp_min_start = min(int(temp_min), int(temp_max))
            temp_min_end = max(int(temp_min), int(temp_max))
            plot_url = plot_prob_dynamic_bar(temp_max, temp_min_start, temp_min_end)
        else:
            # Giá trị mặc định khi GET
            default_max = int(X['temp_max'].mean())
            default_min = int(X['temp_min'].mean())
            plot_url = plot_prob_dynamic_bar(default_max, default_min, default_max)
    except Exception as e:
        ket_qua = f"Lỗi xử lý: {e}"
        plot_url = ""

    return render_template('index.html',
                        ket_qua=ket_qua,
                        prob=prob,
                        temp_max=temp_max,
                        temp_min=temp_min,
                        plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=False)
