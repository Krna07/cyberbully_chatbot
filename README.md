# Chatbot Service

Place your trained chatbot model files in this directory:

```
chatbot-service/
├── app.py                    ← FastAPI service
├── requirements.txt          ← Python dependencies
├── chatbot_model.pkl         ← Your trained chatbot model
├── chatbot_vectorizer.pkl    ← Your trained vectorizer
└── (any other pkl files)     ← Additional model files
```

## Running Locally

```bash
cd chatbot-service
pip install -r requirements.txt
python -m uvicorn app:app --port 8001
```

## API Endpoint

POST /chat
```json
{
  "message": "I am feeling stressed",
  "context": []
}
```

Response:
```json
{
  "response": "I understand you're feeling stressed...",
  "model_used": "trained_chatbot_model"
}
```
