from flask import Flask, render_template
from random import randint

app = Flask(__name__)

@app.route('/')
def index():
    line = "M 0 150 L 50 100 L 100 50 L 150 100 L 200 100 L 250 50 L 300 75"
    return render_template('index.html', line=line, ecg_h=100, ecg_w=500)

if __name__ == '__main__':
    app.run(debug=True)

