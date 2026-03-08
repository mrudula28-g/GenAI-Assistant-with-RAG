from flask import Flask, request, jsonify, render_template
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os

app = Flask(__name__)

client = Groq(
    api_key="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"
)

SIMILARITY_THRESHOLD = 0.3
TOP_K = 3
MAX_HISTORY = 5

conversation_history = {}

def chunk_document(text, chunk_size=80):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks


def generate_embedding(text):
    words = text.lower().split()
    vector = np.zeros(100)

    for word in words:
        index = hash(word) % 100
        vector[index] += 1

    return vector


with open("docs.json") as f:
    documents = json.load(f)


vector_store = []

for doc in documents:

    chunks = chunk_document(doc["content"])

    for chunk in chunks:

        embedding = generate_embedding(chunk)

        vector_store.append({
            "text": chunk,
            "embedding": embedding
        })


def retrieve_chunks(query_embedding):

    similarities = []

    for item in vector_store:

        score = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        similarities.append((score, item["text"]))

    similarities.sort(reverse=True)

    return similarities[:TOP_K]


def build_prompt(context, history, question):

    history_text = ""

    for pair in history:
        history_text += f"User: {pair[0]}\nAssistant: {pair[1]}\n"

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the information provided in the context.

If the answer is not present in the context say:
"I don't have enough information to answer that."

Context:
{context}

Conversation History:
{history_text}

User Question:
{question}

Answer:
"""

    return prompt


def get_llm_response(prompt):

    try:

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=200
        )

        answer = response.choices[0].message.content

        return answer, 0

    except Exception as e:

        print("LLM Error:", e)

        return "AI service temporarily unavailable.", 0


def update_history(session_id, user_msg, assistant_msg):

    if session_id not in conversation_history:
        conversation_history[session_id] = []

    conversation_history[session_id].append((user_msg, assistant_msg))

    conversation_history[session_id] = conversation_history[session_id][-MAX_HISTORY:]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():

    try:

        data = request.json

        session_id = data.get("sessionId")
        message = data.get("message")

        if not message:
            return jsonify({"error": "Message is required"}), 400

        query_embedding = generate_embedding(message)

        retrieved = retrieve_chunks(query_embedding)

        if not retrieved:
            return jsonify({"reply": "No relevant documents found."})

        if retrieved[0][0] < SIMILARITY_THRESHOLD:
            return jsonify({
                "reply": "I don't have enough information in the documents.",
                "retrievedChunks": 0,
                "tokensUsed": 0
            })

        context = "\n".join([c[1] for c in retrieved])

        history = conversation_history.get(session_id, [])

        prompt = build_prompt(context, history, message)

        reply, tokens_used = get_llm_response(prompt)

        update_history(session_id, message, reply)

        return jsonify({
            "reply": reply,
            "tokensUsed": tokens_used,
            "retrievedChunks": len(retrieved)
        })

    except Exception as e:

        print("API Error:", e)

        return jsonify({
            "error": "Internal server error"
        }), 500


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
