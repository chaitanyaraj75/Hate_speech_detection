from flask import Flask,request, make_response,jsonify
from flask_cors import CORS
import os 

import sklearn
print(sklearn.__version__)

import joblib
print(joblib.__version__)

# --- 1. Load pipeline ---
from pathlib import Path

# Try a local relative path first, then fall back to the original absolute path used by the project
local_path = Path(__file__).parent / "resources" / "hate_speech_pipeline.pkl"
abs_path = Path(r"C:\Users\Ashis\.vscode\extensions\.vscode\.vscode\backend_tut\flask_backend\backend\resources\hate_speech_pipeline.pkl")

pipeline = None
for p in (local_path, abs_path):
    try:
        if p.exists():
            pipeline = joblib.load(p)
            print(f"Loaded pipeline from: {p}")
            break
    except Exception as e:
        print(f"Failed to load pipeline from {p}: {e}")

if pipeline is None:
    print("Warning: pipeline not found. '/predict' endpoint will return an error until the model file is provided at 'backend/resources/hate_speech_pipeline.pkl' or the path in app.py is updated.")

app = Flask(__name__)

# Enable CORS for all routes
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# --- 4. Define prediction route ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON request
        data = request.get_json(force=True)
        if 'tweet' not in data:
            return jsonify({'error': 'Missing "tweet" in request'}), 400

        tweet_text = [data['tweet']]  # must be a list for the pipeline
        prediction = pipeline.predict(tweet_text)[0]

        # Return prediction as JSON
        return jsonify({'tweet': data['tweet'], 'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route("/")
def hello_world():
    return"<p>Hello world</p>"

@app.route("/health")
def health_check():
    return "<p>Health check OK</p>"

@app.route("/greet/<name>")
def greet(name):
    return f"<p>Hello, {name}!</p>"

@app.route("/add/<int:a>/<int:b>")
def add(a, b):
    return f"<p>The sum of {a} and {b} is {a + b}</p>"

@app.route("/handle_url_params")
def handle_url_params():
    #/handle_url_params?param1=value1&param2=value2
    name = request.args.get("param1", "default_value1")
    greeting = request.args.get("param2", "default_value2")
    return f"<p>{name}, {greeting}</p>"

@app.route("/hello",methods=["POST", "GET", "PUT"])
def hello():
    if request.method == "POST":
        #curl -X POST http://127.0.0.1:3001/hello
        return "<p>Hello from POST</p>"
    elif request.method == "GET":
        #curl http://127.0.0.1:3001/hello
        return "<p>Hello from GET</p>"
    else:
        return "<p>Hello from PUT</p>"
    
@app.route("/xyz")
def xyz():
    return "<p>xyz</p>",200

@app.route("/xyz2")
def xyz2():
    response = make_response("<p>xyz2</p>")
    response.status_code = 201
    response.headers["content-type"] = "text/plain"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=3001,debug=True)
