# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
from datetime import datetime, timedelta
import json
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading models, useful for persistence
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from google.oauth2 import id_token
from google.auth.transport import requests as grequests


app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing your frontend to access it

# --- Data Definitions ---
pulses = [
    {'name': 'Toor Dal', 'basePrice': 90, 'volatility': 0.08, 'seasonal_amplitude': 5},
    {'name': 'Moong Dal', 'basePrice': 75, 'volatility': 0.06, 'seasonal_amplitude': 4},
    {'name': 'Masoor Dal', 'basePrice': 80, 'volatility': 0.07, 'seasonal_amplitude': 4.5},
    {'name': 'Chana Dal', 'basePrice': 65, 'volatility': 0.05, 'seasonal_amplitude': 3},
]

vegetables = [
    {'name': 'Potatoes', 'basePrice': 25, 'volatility': 0.15, 'seasonal_amplitude': 8},
    {'name': 'Onions', 'basePrice': 30, 'volatility': 0.20, 'seasonal_amplitude': 10},
    {'name': 'Tomatoes', 'basePrice': 40, 'volatility': 0.25, 'seasonal_amplitude': 12},
    {'name': 'Cabbage', 'basePrice': 20, 'volatility': 0.10, 'seasonal_amplitude': 6},
]

seasons = [
    {'name': 'Spring', 'months': 'Mar-May', 'impact': 'harvest'},
    {'name': 'Summer', 'months': 'Jun-Aug', 'impact': 'heat_stress'},
    {'name': 'Monsoon', 'months': 'Sep-Nov', 'impact': 'planting'},
    {'name': 'Winter', 'months': 'Dec-Feb', 'impact': 'storage'},
]

# Store trained ML models and their scalers
ml_models = {}
feature_scalers = {}
target_scalers = {}

# --- Synthetic Data Generation for ML Training ---
def generate_synthetic_historical_data(item_data, start_date_str, num_days=365*2): # 2 years of historical data
    """Generates synthetic historical price data for a given item."""
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    data = []
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        day_of_year = current_date.timetuple().tm_yday # 1-366

        # Base price with a slight upward trend over time
        trend_component = item_data['basePrice'] + (i / num_days) * (item_data['basePrice'] * 0.1) # 10% increase over 2 years

        # Seasonal component (sinusoidal, peaking around mid-year for summer produce, etc.)
        # Adjust phase for different items if needed, e.g., for winter crops
        seasonal_component = item_data['seasonal_amplitude'] * np.sin(2 * np.pi * day_of_year / 365.25 + np.pi/2)

        # Volatility/noise
        noise = (random.random() - 0.5) * item_data['volatility'] * item_data['basePrice'] * 2

        price = trend_component + seasonal_component + noise
        price = max(1, round(price, 2)) # Ensure price is not negative

        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'price': price,
            'day_index': i,
            'day_of_year': day_of_year,
            'sin_day_of_year': np.sin(2 * np.pi * day_of_year / 365.25),
            'cos_day_of_year': np.cos(2 * np.pi * day_of_year / 365.25)
        })
    return data

# --- ML Model Training ---
def train_commodity_models():
    """Trains a linear regression model for each commodity."""
    all_commodities = pulses + vegetables
    today = datetime.now()
    # Start historical data from 2 years ago
    start_historical_date = (today - timedelta(days=365*2)).strftime('%Y-%m-%d')

    for item_data in all_commodities:
        item_name = item_data['name']
        print(f"Training model for {item_name}...")

        # Generate synthetic historical data for training
        historical_data = generate_synthetic_historical_data(item_data, start_historical_date)

        # Prepare features (X) and target (y)
        X = []
        y = []
        for entry in historical_data:
            # Features: day_index, sin(day_of_year), cos(day_of_year)
            X.append([entry['day_index'], entry['sin_day_of_year'], entry['cos_day_of_year']])
            y.append(entry['price'])

        X = np.array(X)
        y = np.array(y).reshape(-1, 1) # Reshape for scaler

        # Scale features and target
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()

        X_scaled = feature_scaler.fit_transform(X)
        y_scaled = target_scaler.fit_transform(y)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_scaled, y_scaled)

        # Store the trained model and scalers
        ml_models[item_name] = model
        feature_scalers[item_name] = feature_scaler
        target_scalers[item_name] = target_scaler
        print(f"Model for {item_name} trained successfully.")

# --- Prediction Logic using Trained ML Models ---
def generate_chart_data_ml(selected_item_data, forecast_date_str, prediction_period):
    """
    Generates historical and forecast data using the trained ML model.
    The historical part will be a subset of the data used for training.
    The forecast part will use the trained model to predict.
    """
    item_name = selected_item_data['name']
    model = ml_models.get(item_name)
    feature_scaler = feature_scalers.get(item_name)
    target_scaler = target_scalers.get(item_name)

    if not model or not feature_scaler or not target_scaler:
        print(f"ML model for {item_name} not found. Falling back to simulation.")
        # Fallback to original simulation if model not found (shouldn't happen after training)
        return generate_chart_data_simulation(selected_item_data, forecast_date_str, prediction_period)

    today = datetime.now()
    forecast_dt = datetime.strptime(forecast_date_str, '%Y-%m-%d')
    chart_data = []

    # Get the last 30 days of historical data (from the synthetic data used for training)
    # This assumes we have enough historical data generated
    start_historical_data_for_chart = (today - timedelta(days=30))
    history_training_start_date = datetime.strptime((today - timedelta(days=365*2)).strftime('%Y-%m-%d'), '%Y-%m-%d')
    current_day_index = (start_historical_data_for_chart - history_training_start_date).days # Index from start of training data

    for i in range(31): # 30 historical days + current day
        current_date = start_historical_data_for_chart + timedelta(days=i)
        day_of_year = current_date.timetuple().tm_yday
        day_index_for_prediction = current_day_index + i

        # Predict historical price (use model for consistency, or use actual generated data if available)
        features = np.array([[day_index_for_prediction, np.sin(2 * np.pi * day_of_year / 365.25), np.cos(2 * np.pi * day_of_year / 365.25)]])
        features_scaled = feature_scaler.transform(features)
        predicted_price_scaled = model.predict(features_scaled)
        predicted_price = target_scaler.inverse_transform(predicted_price_scaled)[0][0] # Corrected variable name
        predicted_price = max(1, round(predicted_price, 2))

        chart_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'price': predicted_price,
            'type': 'Historical'
        })

    # Determine forecast days based on predictionPeriod
    forecast_days_map = {
        '1 Week': 7, '2 Weeks': 14, '1 Month': 30,
        '3 Months': 90, '6 Months': 180, '1 Year': 365
    }
    forecast_days = forecast_days_map.get(prediction_period, 7)

    # Generate forecast data using the ML model
    for i in range(1, forecast_days + 1): # Start from 1 day after the forecast_dt
        current_date = forecast_dt + timedelta(days=i)
        day_of_year = current_date.timetuple().tm_yday
        # Calculate day_index relative to the start of the ML model's training data
        day_index_for_prediction = (current_date - history_training_start_date).days

        features = np.array([[day_index_for_prediction, np.sin(2 * np.pi * day_of_year / 365.25), np.cos(2 * np.pi * day_of_year / 365.25)]])
        features_scaled = feature_scaler.transform(features)
        predicted_price_scaled = model.predict(features_scaled)
        predicted_price = target_scaler.inverse_transform(predicted_price_scaled)[0][0] # Corrected variable name
        predicted_price = max(1, round(predicted_price, 2))

        chart_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'price': predicted_price,
            'type': 'Forecast'
        })
    return chart_data

# Original simulation fallback (in case ML model fails or isn't trained)
def generate_chart_data_simulation(selected_item, selected_date_str, period):
    """Generates simulated historical and forecast data for the chart (fallback)."""
    if not selected_item or not selected_date_str:
        return []

    today = datetime.now()
    forecast_dt = datetime.strptime(selected_date_str, '%Y-%m-%d')
    data = []

    # Simulate historical data for the past 30 days
    for i in range(30, -1, -1):
        date = today - timedelta(days=i)
        price = selected_item['basePrice'] * (1 + (random.random() - 0.5) * selected_item['volatility'])
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(price, 2),
            'type': 'Historical'
        })

    # Determine forecast days based on predictionPeriod
    forecast_days = 0
    if period == '1 Week':
        forecast_days = 7
    elif period == '2 Weeks':
        forecast_days = 14
    elif period == '1 Month':
        forecast_days = 30
    elif period == '3 Months':
        forecast_days = 90
    elif period == '6 Months':
        forecast_days = 180
    elif period == '1 Year':
        forecast_days = 365
    else:
        forecast_days = 7 # Default

    # Simulate forecast data for the selected prediction period
    last_historical_price = data[-1]['price']
    for i in range(forecast_days + 1):
        date = forecast_dt + timedelta(days=i)
        # Simple linear trend for forecast, with some noise
        trend_factor = (forecast_dt - today).days / 30
        forecast_price = last_historical_price * (1 + (trend_factor * 0.05) + (random.random() - 0.5) * selected_item['volatility'] * 0.5)
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(forecast_price, 2),
            'type': 'Forecast'
        })
    return data

def generate_current_prices_py():
    """Simulates current market prices."""
    all_produce = pulses + vegetables
    current_prices = []
    for p in all_produce:
        price = p['basePrice'] * (1 + (random.random() - 0.5) * p['volatility'] * 0.5)
        current_prices.append({
            'item': p['name'],
            'price': round(price, 2)
        })
    return current_prices

async def get_llm_insight_py(item, forecasted_price, prediction_period, selected_season_data):
    """Calls Gemini API for market insights."""
    seasonal_text = ''
    if selected_season_data and selected_season_data.get('name') != '':
        seasonal_text = f" During {selected_season_data.get('name')} season ({selected_season_data.get('months')}), expect {selected_season_data.get('impact').replace('_', ' ')} related price variations."

    prompt = f"Advanced market analysis indicates {item} prices will show {'upward' if forecasted_price > 0 else 'downward'} momentum. Key factors include seasonal patterns, supply chain dynamics, weather conditions, and regional demand fluctuations.{seasonal_text} Current market volatility is calculated based on historical data. Monitor government policies, export demands, and storage costs for additional price impacts."

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}
    api_key = "" # Leave empty for Canvas environment, or add your key for local testing.
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        result = response.json()

        if result.get('candidates') and len(result['candidates']) > 0 and \
           result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts') and \
           len(result['candidates'][0]['content']['parts']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print("LLM response structure unexpected:", result)
            return "No specific insights available at this time."
    except requests.exceptions.RequestException as e:
        print(f"Error calling LLM API: {e}")
        return "Failed to generate market insights due to an API error."
    except Exception as e:
        print(f"An unexpected error occurred in LLM call: {e}")
        return "Failed to generate market insights due to an unexpected error."


# --- Flask Routes ---
@app.route('/predict', methods=['POST'])
async def predict():
    data = request.json
    category = data.get('category')
    item = data.get('item')
    forecast_date_str = data.get('forecastDate')
    prediction_period = data.get('predictionPeriod')
    selected_season_name = data.get('selectedSeason')

    if not all([category, item, forecast_date_str, prediction_period, selected_season_name]):
        return jsonify({'error': 'Missing required parameters'}), 400
    selected_produce_list = pulses if category == 'pulses' else vegetables
    selected_item_data = next((p for p in selected_produce_list if p['name'] == item), None)
    selected_season_data = next((s for s in seasons if s['name'] == selected_season_name), None)

    if not selected_item_data:
     return jsonify({'error': 'Selected item not found.'}), 404
        
    # Use the ML model for chart data generation
    chart_data = generate_chart_data_ml(selected_item_data, forecast_date_str, prediction_period)

    # Find the forecasted price for the exact forecastDate
    specific_forecast_point = next((d for d in chart_data if d['date'] == forecast_date_str), None)
    final_forecast_price = specific_forecast_point['price'] if specific_forecast_point else chart_data[-1]['price']

    forecast_result = {
        'item': item,
        'date': forecast_date_str,
        'price': final_forecast_price,
        'unit': '₹/kg',
        'trend': 'Upward' if final_forecast_price > selected_item_data['basePrice'] else 'Downward',
        'confidence': f"{random.randint(70, 90)}%", # Simulate confidence
        'season': selected_season_data['name'] if selected_season_data else 'Not specified',
        'seasonalImpact': selected_season_data['impact'] if selected_season_data else 'neutral'
    }

    # Get LLM insight
    llm_insight = await get_llm_insight_py(item, final_forecast_price, prediction_period, selected_season_data)

    response_data = {
        'forecastResult': forecast_result,
        'llmInsight': llm_insight,
        'chartData': chart_data,
        'currentPrices': generate_current_prices_py() # Also send current prices for initial display
    }

    return jsonify(response_data)

@app.route('/')
def home():
    return render_template('index.html')

FAQS = {
    "what is agriprice forecast?": "AgriPrice Forecast is an AI-powered platform that predicts agricultural commodity prices using advanced machine learning and data analytics.",
    "how do you predict prices?": "We use machine learning models trained on historical price data, seasonality, and current market trends to forecast crop prices.",
    "who predicts the prices?": "Prices are predicted by our AI models, which are built and maintained by the project developer using scientific forecasting techniques.",
    "what is ai assistant?": "The AI assistant answers your questions about predictions, data, and using this website, powered by a large language model.",
    "how to use this website?": "Select category, item, season, forecast date, and prediction period. Then click 'Predict' to get price forecasts and insights.",
    "how to get price prediction?": "Fill in the form with your desired crop and period, then press 'Predict' to see analytical results and forecast charts.",
    "what do the charts show?": "The charts compare past price history with forecasted prices, helping you visualize trends and make informed decisions.",
    "is this for farmers or traders?": "This site helps both farmers and traders to anticipate price changes and plan better, with data-driven forecasts for key commodities.",
    "what is seasonal impact?": "Seasonal impact shows how specific seasons (like monsoon or harvest) affect price fluctuations for crops and vegetables.",
    "how accurate are the predictions?": "Our predictions are based on historical and current data trends; while not guarantees, they provide a valuable data-driven estimate.",
    "do you support all commodities?": "Our demo supports key pulses and vegetables such as Toor Dal, Moong Dal, Potatoes, and Onions. More crops can be added as needed.",
    "who developed this website?": "This website was developed as an AI-ML student project to help people make smarter agricultural marketing decisions.",
    "how can i get help?": "Use the AI assistant chat at the bottom right, or read the info cards on each section for help.",
    "what technology powers this website?": "The platform uses Flask (Python), JavaScript, and AI APIs like OpenAI or Google Gemini for real-time chat and predictions."
}
def match_faq(question):
    q_clean = question.strip().lower()
    for faqp, answer in FAQS.items():
        if faqp in q_clean:
            return answer
    return None

@app.route('/ai_assistant', methods=['POST'])
def ai_assistant():
    data = request.json
    question = data.get('question', '').strip()
    faq_answer = match_faq(question)
    if faq_answer:
        return jsonify({"answer": faq_answer})
    else:
        return jsonify({"answer": "Sorry, I do not have an answer for that question yet. Please try asking about the website features or usage."})


# Initial setup when the Flask app starts
if __name__ == '__main__':
    # Train ML models on startup
    train_commodity_models()
    # To run this Flask app:
    # 1. Make sure you have Flask, Flask-CORS, numpy, scikit-learn installed:
    #    pip install Flask Flask-Cors requests numpy scikit-learn
    # 2. Save this file as app.py
    # 3. Run from your terminal: python app.py
    #    It will typically run on http://127.0.0.1:5000/
    app.run(debug=True)
