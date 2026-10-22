"""Flask frontend for the clustering models.   Run: python app.py  ->  http://127.0.0.1:5000"""
import pandas as pd
from flask import Flask, render_template_string, request
from inference import predict_all, COLS

app = Flask(__name__)

PAGE = """
<!doctype html><title>Cluster Predictor</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
 input,button{padding:8px;margin:4px 0;width:100%;box-sizing:border-box}
 button{background:#2563eb;color:#fff;border:0;border-radius:6px;cursor:pointer}
 table{border-collapse:collapse;width:100%;margin-top:16px}
 td,th{border:1px solid #ddd;padding:8px;text-align:left} th{background:#f3f4f6}
 .noise{color:#b91c1c;font-weight:600}
</style>
<h2>Cluster Predictor</h2>
<form method="post">
 {% for c in cols %}
  <label>{{ c.title() }}</label>
  <input type="number" step="any" name="{{ c }}" value="{{ values.get(c,'') }}" required>
 {% endfor %}
 <button>Predict cluster</button>
</form>
{% if result %}
<table><tr><th>Algorithm</th><th>Cluster</th></tr>
{% for algo, lab in result.items() %}
 <tr><td>{{ algo }}</td>
 <td>{% if lab == -1 %}<span class="noise">Noise / outlier</span>{% else %}Cluster {{ lab }}{% endif %}</td></tr>
{% endfor %}</table>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result, values = None, {}
    if request.method == "POST":
        values = {c: float(request.form[c]) for c in COLS}
        row = predict_all(pd.DataFrame([values])).iloc[0].drop(COLS)
        result = {k: int(v) for k, v in row.items()}
    return render_template_string(PAGE, cols=COLS, result=result, values=values)

if __name__ == "__main__":
    app.run(debug=True)