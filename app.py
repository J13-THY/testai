import os
import time
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

SYSTEM_PROMPT = """You are the AI-CRIS Assistant — the AI Course Recommendation and Admissions Information System for Kyambogo University, Uganda.

Your role is to help prospective students with:
- Course recommendations
- Entry requirements
- Tuition fees guidance
- Career pathways
- Admissions guidance
- Programme comparisons

Be professional, concise, and helpful.
If unsure, direct students to the Admissions Division.
"""

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    # Keep only recent messages for speed
    messages = data.get("messages", [])[-6:]

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    try:
        start = time.time()

        resp = client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=api_messages,
            max_tokens=150,
            temperature=0.3
        )

        reply = resp.choices[0].message.content

        end = time.time()
        print(f"Response time: {end - start:.2f} seconds")

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)