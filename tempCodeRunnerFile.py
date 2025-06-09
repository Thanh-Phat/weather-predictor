from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # Dùng backend không GUI
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ----- Tiền xử lý dữ liệu -----
df = pd.read_csv("weather_data_2025_5000.csv")
df['weather'] = df['weather'].str.lower()
filtered_df = df[df['weather'].isin(['rain', 'drizzle', 'sun', 'sunny'])].copy()
filtered_df['is_rain'] = filtered_df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)

X = filtered_df[['temp_max', 'temp_min']]
y = filtered_df['is_rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# ----- Hàm dự đoán -----
def du_doan_thoi_tiet(temp_max, temp_min):
    input_data = pd.DataFrame([[temp_max, temp_min]], columns=['temp_max', 'temp_min'])
    prob = model.predict_proba(input_data)[0][1]
    label = 1 if prob >= 0.5 else 0
    return label, prob

# ----- Hàm vẽ biểu đồ -----
def plot_prob_dynamic_bar(temp_min_input, temp_max_input):
    temp_min_values = list(range(int(temp_min_input), int(temp_max_input) + 1))
    probabilities = []

    for temp_min in temp_min_values:
        input_data = pd.DataFrame([[temp_max_input, temp_min]], columns=['temp_max', 'temp_min'])
        prob = model.predict_proba(input_data)[0][1]
        probabilities.append(prob * 100)  # phần trăm

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(temp_min_values, probabilities, color='orange')
    ax.set_xlabel('Nhiệt độ thấp nhất (temp_min)')
    ax.set_ylabel('Xác suất mưa (%)')
    ax.set_title(f'Xác suất mưa khi temp_max = {temp_max_input:.1f}')
    ax.set_ylim(0, 100)
    ax.set_xticks(temp_min_values)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return f'data:image/png;base64,{image_base64}'

# ----- Route chính -----
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        temp_max = float(request.form['temp_max'])
        temp_min = float(request.form['temp_min'])
        label, prob = du_doan_thoi_tiet(temp_max, temp_min)
        ket_qua = "☔ RAIN" if label == 1 else "☀️ SUN"
        plot_url = plot_prob_dynamic_bar(temp_min, temp_max)
    else:
        temp_max = temp_min = prob = ket_qua = None
        plot_url = plot_prob_dynamic_bar(X['temp_min'].mean(), X['temp_max'].mean())

    return render_template(
        'index.html',
        ket_qua=ket_qua,
        prob=prob,
        temp_max=temp_max,
        temp_min=temp_min,
        plot_url=plot_url
    )

# ----- Chạy app -----
if __name__ == '__main__':
    app.run(debug=True, threaded=False, use_reloader=False)
