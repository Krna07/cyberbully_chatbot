from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = Path(__file__).parent
chatbot_model = None
chatbot_vectorizer = None
chatbot_responses = None

try:
    with open(model_path / "mental_support_chatbot.pkl", "rb") as f:
        chatbot_model = pickle.load(f)
    print("✓ Chatbot model loaded (NearestNeighbors)")
except Exception as e:
    print(f"✗ Chatbot model error: {e}")

try:
    with open(model_path / "mental_support_vectorizer.pkl", "rb") as f:
        chatbot_vectorizer = pickle.load(f)
    print("✓ Chatbot vectorizer loaded (TF-IDF, vocab=5000)")
except Exception as e:
    print(f"✗ Chatbot vectorizer error: {e}")

try:
    with open(model_path / "mental_support_responses.pkl", "rb") as f:
        chatbot_responses = pickle.load(f)
    print(f"✓ Chatbot responses loaded ({len(chatbot_responses)} entries)")
except Exception as e:
    print(f"✗ Chatbot responses error: {e}")


class ChatInput(BaseModel):
    message: str
    context: list = []


def get_model_response(message: str) -> str:
    """Use NearestNeighbors to find the best matching response."""
    vectorized = chatbot_vectorizer.transform([message])
    distances, indices = chatbot_model.kneighbors(vectorized, n_neighbors=1)
    distance = distances[0][0]

    # If distance is too large (no good match), fall back to rule-based
    if distance > 0.95:
        return get_fallback_response(message)

    return chatbot_responses.iloc[indices[0][0]]


def get_fallback_response(message: str) -> str:
    msg = message.lower()

    if any(w in msg for w in ['stress', 'anxious', 'anxiety', 'worried', 'overwhelm']):
        return ("I understand you're feeling stressed. Try taking 5 deep breaths — "
                "inhale for 4 counts, hold for 4, exhale for 4. "
                "Would you like more stress management tips?")

    if any(w in msg for w in ['sad', 'depressed', 'unhappy', 'cry', 'hopeless', 'lonely']):
        return ("I'm sorry you're feeling this way. It's okay to feel sad sometimes. "
                "Talking to someone you trust can really help. Would you like some resources?")

    if any(w in msg for w in ['bullied', 'bully', 'harassed', 'threatened', 'cyberbully']):
        return ("I'm sorry to hear you're being bullied. You're not alone. "
                "Please consider reporting it using our detection tool and talking to a trusted adult or authority.")

    if any(w in msg for w in ['help', 'support', 'advice', 'what can']):
        return ("I'm here to help! You can: 1) Use our detection tool to analyze messages, "
                "2) Report bullying to authorities, 3) Talk to a counselor. What would you like to do?")

    if any(w in msg for w in ['hi', 'hello', 'hey', 'good morning', 'good evening']):
        return ("Hello! I'm your mental support assistant. "
                "I'm here to help you manage stress and deal with cyberbullying. How are you feeling today?")

    if any(w in msg for w in ['thank', 'thanks', 'appreciate']):
        return "You're welcome! Remember, you're not alone. I'm always here if you need support. 💙"

    if any(w in msg for w in ['angry', 'anger', 'frustrated', 'mad', 'furious']):
        return ("It's natural to feel angry sometimes. Try stepping away for a moment, "
                "take a few deep breaths, and give yourself space to cool down. Want to talk about what's bothering you?")

    return ("I'm here to support you. Can you tell me more about what you're going through? "
            "I can help with stress management and cyberbullying resources.")


@app.post("/chat")
async def chat(input_data: ChatInput):
    try:
        if chatbot_model and chatbot_vectorizer and chatbot_responses is not None:
            response = get_model_response(input_data.message)
            return {"response": response, "model_used": "mental_support_trained_model"}
        else:
            return {
                "response": get_fallback_response(input_data.message),
                "model_used": "fallback"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "chatbot_model": chatbot_model is not None,
        "chatbot_vectorizer": chatbot_vectorizer is not None,
        "chatbot_responses": chatbot_responses is not None,
        "response_count": len(chatbot_responses) if chatbot_responses is not None else 0
    }
