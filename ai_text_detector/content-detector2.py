import requests

url = "https://ai-content-detector2.p.rapidapi.com/analyzePatterns"

payload = {
    "text": "Bottom line: your request shape is fine; the model itself is broken on the server..."
}
headers = {
    "x-rapidapi-key": "api_key",
    "x-rapidapi-host": "ai-content-detector2.p.rapidapi.com",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
response_data = response.json()

# Ensure it's a list
if isinstance(response_data, list) and len(response_data) > 0:
    aggregated = response_data[0]["aggregated"]
    confidence = aggregated["aggregated_confidence"]
    prediction = aggregated["aggregated_prediction"]

    # Compute AI score
    ai_score = confidence if prediction == "AI-generated" else 1 - confidence
    human_score = 1 - ai_score

    print(f"AI Score: {ai_score:.2%}")
    print(f"Human Score: {human_score:.2%}")

else:
    # Print raw response if it's not the expected format
    print("Unexpected response format:", response_data)
