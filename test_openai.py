
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

response = openai.chat.Completions.create(
   model="gpt-3.5-turbo",
   messages=[{"role": "user", "content": "Say hello!"}
   ]
)

print(response.choices[0].message.content)
