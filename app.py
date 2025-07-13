from flask import Flask, render_template, request
from weather_logic import du_doan_thoi_tiet, plot_prob_dynamic_bar

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    temp = humidity = prob = ket_qua = plot_url = None

    try:
        if request.method == 'POST':
            temp = float(request.form['temp'])
            humidity = float(request.form['humidity'])

            if temp > 50 or humidity < 10 or humidity > 100:
                raise ValueError("Giá trị nằm ngoài phạm vi!")

            label, prob = du_doan_thoi_tiet(humidity, temp)
            ket_qua = " Có mưa" if label else " Không mưa"

            hum_start = max(10, int(humidity) - 5)
            hum_end = min(100, int(humidity) + 5)
            plot_url = plot_prob_dynamic_bar(temp, hum_start, hum_end)

        else:
            temp = 26
            humidity = 60
            plot_url = plot_prob_dynamic_bar(temp, 55, 65)

    except Exception as e:
        ket_qua = f"Lỗi: {e}"
        plot_url = ""

    return render_template('index.html',
                            ket_qua=ket_qua,
                            prob=prob,
                            temp=temp,
                            humidity=humidity,
                            plot_url=plot_url)
    
if __name__ == '__main__':
    app.run(debug=True)

