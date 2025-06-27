from flask import Flask, render_template, request
from weather_logic import du_doan_thoi_tiet, plot_prob_dynamic_bar

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    temp_max = temp_min = prob = ket_qua = plot_url = None
    try:
        if request.method == 'POST':
            temp_max = float(request.form['temp_max'])
            temp_min = float(request.form['temp_min'])
            
            if temp_max > 40 or temp_min < 20:
                raise ValueError("Nhiệt độ ngoài khoảng cho phép (-10 đến 45 độ C)")

            label, prob = du_doan_thoi_tiet(temp_max, temp_min)
            ket_qua = "RAIN ☔" if label == 1 else "SUN ☀️"

            temp_min_start = min(int(temp_min), int(temp_max))
            temp_min_end = max(int(temp_min), int(temp_max))
            plot_url = plot_prob_dynamic_bar(temp_max, temp_min_start, temp_min_end)
        else:
            plot_url = plot_prob_dynamic_bar(30, 25, 30)
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
    app.run(debug=True)
