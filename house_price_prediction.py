from flask import Flask, request, render_template_string
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
    <style>
        body {
            background: linear-gradient(120deg, #1f4037, #99f2c8);
            font-family: Arial, sans-serif;
        }
        .box {
            width: 420px;
            margin: 80px auto;
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        }
        h2 {
            text-align: center;
            color: #1f4037;
        }
        input {
            width: 100%;
            padding: 10px;
            margin: 6px 0;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        button {
            width: 100%;
            padding: 12px;
            background: #1f4037;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background: #16312b;
        }
        .result {
            margin-top: 15px;
            padding: 12px;
            background: #e7f8f1;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            color: #1f4037;
        }
    </style>
</head>

<body>
    <div class="box">
        <h2>🏠 House Price Prediction</h2>

        <form method="POST">
            <input type="number" step="any" name="f1" placeholder="Feature 1" required>
            <input type="number" step="any" name="f2" placeholder="Feature 2" required>
            <input type="number" step="any" name="f3" placeholder="Feature 3" required>
            <input type="number" step="any" name="f4" placeholder="Feature 4" required>
            <input type="number" step="any" name="f5" placeholder="Feature 5" required>
            <input type="number" step="any" name="f6" placeholder="Feature 6" required>
            <input type="number" step="any" name="f7" placeholder="Feature 7" required>
            <input type="number" step="any" name="f8" placeholder="Feature 8" required>
            <input type="number" step="any" name="f9" placeholder="Feature 9" required>
            <input type="number" step="any" name="f10" placeholder="Feature 10" required>
            <button type="submit">Predict</button>
        </form>

        {% if prediction is not none %}
        <div class="result">
            Predicted Price: {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        values = [float(request.form[f"f{i}"]) for i in range(1, 11)]
        result = model.predict(np.array(values).reshape(1, -1))[0]
        prediction = round(result, 2)

    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
