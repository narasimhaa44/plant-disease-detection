import requests

def get_weather(city_name, api_key):
    base_url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": api_key,
        "q": city_name
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code != 200:
            return {"error": f"Failed to fetch weather data: {data}"}

        current = data["current"]
        weather = {
            "temperature": current["temp_c"],
            "humidity": current["humidity"],
            "conditions": current["condition"]["text"],
            "rain_1h_mm": current.get("precip_mm", 0)
        }
        return weather

    except Exception as e:
        return {"error": str(e)}
