from flask import Flask, render_template, request, Response
import pandas as pd
from utils import generate_features, train_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    period = request.form['period']

    if uploaded_file.filename != '':
        df = pd.read_csv(uploaded_file)
        df_feat = generate_features(df)
        result_df, img_base64, csv_data = train_and_predict(df_feat, period)

        return render_template('result.html',
                               table=result_df.to_html(index=False),
                               img_data=img_base64,
                               csv_data=csv_data)
    return "ファイルをアップロードしてください"

@app.route('/download', methods=['POST'])
def download():
    csv_data = request.form['csv']
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=prediction.csv"}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

