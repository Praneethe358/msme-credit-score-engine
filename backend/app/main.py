from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.schemas.predict import MSMEPredictionInput, MSMEPredictionOutput
from app.services.model import model_service

app = FastAPI(
    title="MSME Risk Scoring & Eligibility API",
    description="A FastAPI microservice providing machine learning-based credit eligibility and risk scoring for MSMEs.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MSME ML API Service</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Outfit', sans-serif;
                background-color: #0b0f19;
                color: #f3f4f6;
                margin: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: 100vh;
                background-image: radial-gradient(circle at 10% 20%, rgba(30, 41, 59, 0.4) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
            }
            .container {
                text-align: center;
                padding: 3rem;
                background: rgba(17, 24, 39, 0.7);
                backdrop-filter: blur(12px);
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
                max-width: 600px;
                width: 90%;
            }
            h1 {
                font-size: 2.5rem;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
            }
            p {
                color: #9ca3af;
                font-size: 1.1rem;
                line-height: 1.6;
                margin-bottom: 2rem;
            }
            .badge {
                display: inline-block;
                padding: 0.4rem 1rem;
                background: rgba(99, 102, 241, 0.2);
                border: 1px solid rgba(99, 102, 241, 0.3);
                border-radius: 50px;
                color: #a5b4fc;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
            }
            .buttons {
                display: flex;
                gap: 1rem;
                justify-content: center;
            }
            .btn {
                padding: 0.8rem 1.8rem;
                border-radius: 10px;
                font-weight: 600;
                text-decoration: none;
                transition: all 0.3s ease;
                font-size: 1rem;
            }
            .btn-primary {
                background: #6366f1;
                color: #ffffff;
                box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
            }
            .btn-primary:hover {
                background: #4f46e5;
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
            }
            .btn-secondary {
                background: transparent;
                color: #d1d5db;
                border: 1px solid rgba(255, 255, 255, 0.15);
            }
            .btn-secondary:hover {
                background: rgba(255, 255, 255, 0.05);
                color: #ffffff;
                transform: translateY(-2px);
            }
            .footer {
                margin-top: 3rem;
                font-size: 0.8rem;
                color: #4b5563;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <span class="badge">FASTAPI ML SERVICE</span>
            <h1>MSME Risk Engine</h1>
            <p>Welcome to the ML predictor microservice for Micro, Small & Medium Enterprises. This backend evaluates creditworthiness, determines eligibility probabilities, and scores risk indicators.</p>
            <div class="buttons">
                <a href="/docs" class="btn btn-primary">API Documentation</a>
                <a href="/redoc" class="btn btn-secondary">Alternative Specs</a>
            </div>
        </div>
        <div class="footer">
            Powered by FastAPI & scikit-learn
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_version": model_service.model_version,
        "model_loaded": model_service.model is not None
    }

@app.post("/api/predict", response_model=MSMEPredictionOutput)
async def predict_risk(payload: MSMEPredictionInput):
    try:
        prediction = model_service.predict(payload)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
