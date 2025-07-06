from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_car_info(user_input):
    prompt = f"""
You are an information extractor. Given a user's car description, extract only the following:
- brand
- year
- transmission
- miles_driven (odometer value in miles)

 Do NOT include any fuel type (like petrol, diesel, electric). Just ignore it if mentioned.

Return the result as a JSON object with keys: brand, year, transmission, miles_driven.

Input: "{user_input}"
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print("Failed to parse GPT response:", e)
        return None

