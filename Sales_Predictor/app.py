from flask import Flask, render_template, request, send_file
import pandas as pd
import os
from utils import generate_features, train_and_predict, plot_forecast
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    # ファイル取得
    file = request.files['file']
    if not file:
        return "ファイルがアップロードされていません", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("static", filename)
    file.save(filepath)

    # CSV読み込み
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])

    # 予測期間（1, 7, 30, 90, 180, 365）
    period = int(request.form['period'])

    # 特徴量作成＋予測
    df_result = train_and_predict(df, period)

    # グラフ画像を生成・保存
    plot_forecast(df_result['Date'], df_result['Sales'])

    # 結果ページへ
    return render_template('result.html', tables=[df_result.to_html(classes='data')], titles=df_result.columns.values)

@app.route('/download')
def download():
    return send_file('static/forecast.png', as_attachment=True)

@app.route('/details')
def details():
    return render_template('details.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)