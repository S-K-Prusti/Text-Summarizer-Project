# Import libraries
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize FastAPI app
app = FastAPI()

# Model name (Hugging Face pre-trained model)
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="interface/static"), name="static")
templates = Jinja2Templates(directory="interface/templates")


# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": None})


# Summarization route
@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, dialogue: str = Form(...)):
    try:
        # Tokenize input
        inputs = tokenizer(
            dialogue,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=4,
            max_length=128,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": summary, "dialogue": dialogue}
        )

    except Exception as e:
        # Return error gracefully on UI
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "summary": f"⚠️ Internal Error: {str(e)}", "dialogue": dialogue}
        )
