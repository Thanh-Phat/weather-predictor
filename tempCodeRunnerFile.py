from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

app = Flask(__name__)

# ----- Tiền xử lý dữ liệu -----
df = pd.read_csv("weather_dataset.csv")
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

# ----- Vẽ biểu đồ -----
def plot_prob_dynamic_bar(temp_min_input):
    temp_max_mean = X['temp_max'].mean()

    input_data = pd.DataFrame([[temp_max_mean, temp_min_input]], columns=['temp_max', 'temp_min'])
    prob = model.predict_proba(input_data)[0][1]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(['Rain Probability'], [prob], color='skyblue')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Dự đoán khả năng mưa (với temp_max trung bình)')

    # Encode ảnh thành base64
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
        ket_qua = "RAIN ☔" if label == 1 else "SUN ☀️"
        plot_url = plot_prob_dynamic_bar(temp_min)
    else:
        temp_max = temp_min = prob = ket_qua = None
        plot_url = plot_prob_dynamic_bar(X['temp_min'].mean())

    return render_template(
        'index.html',
        ket_qua=ket_qua,
        prob=prob,
        temp_max=temp_max,
        temp_min=temp_min,
        plot_url=plot_url
    )

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
