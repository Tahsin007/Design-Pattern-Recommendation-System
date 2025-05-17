from flask import Flask, render_template, request, jsonify
from model_utils import load_model, get_recommendations_with_diagram
import time

app = Flask(__name__)

# Load the model when the app starts
model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    
    # Get the problem description from the form
    problem_text = request.form['problem']
    
    # Get recommendations from the model
    recommendations = get_recommendations_with_diagram(model, problem_text)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Add processing time to results
    recommendations['processing_time'] = f"{processing_time:.2f} seconds"
    
    return render_template('results.html', results=recommendations)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    start_time = time.time()
    
    # Get the problem description from JSON request
    data = request.get_json()
    problem_text = data['problem']
    
    # Get recommendations from the model
    recommendations = get_recommendations_with_diagram(model, problem_text)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Add processing time to results
    recommendations['processing_time'] = f"{processing_time:.2f} seconds"
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)