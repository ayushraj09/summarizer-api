from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from enum import Enum
import os

# Initialize FastAPI app
app = FastAPI()

# Allow all origins (can be restricted to certain domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains to access your API (update for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.join(os.path.dirname(__file__), "saved_model")
model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
tokenizer = BartTokenizer.from_pretrained(model_dir)

# Enum for summarization levels
class SummaryLevel(str, Enum):
    shortest = "shortest"
    normal = "normal"
    elaborative = "elaborative"

class TextInput(BaseModel):
    text: str
    level: SummaryLevel

@app.post("/summarize/")
def summarize(input_text: TextInput):
    if not input_text.text:
        raise HTTPException(status_code=400, detail="The 'text' field is required and cannot be empty.")
    
    # Split input text into words to check its length
    words = input_text.text.split()
    if len(words) < 10:
        return {
            "summary": input_text.text,
            "message": "Not much to summarize in this email."
        }

    # Tokenize input
    try:
        inputs = tokenizer.encode(input_text.text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in tokenization: {str(e)}")
    
    input_length = len(inputs[0])

    # Set summarization parameters based on level
    if input_text.level == "shortest":
        min_len = max(int(input_length * 0.1), 20)
        max_len = max(int(input_length * 0.2), 50)
        num_beams = 2
        length_penalty = 0.8
    elif input_text.level == "normal":
        min_len = max(int(input_length * 0.2), 40)
        max_len = max(int(input_length * 0.4), 100)
        num_beams = 4
        length_penalty = 1.0
    elif input_text.level == "elaborative":
        min_len = max(int(input_length * 0.3), 60)
        max_len = max(int(input_length * 0.6), 150)
        num_beams = 6
        length_penalty = 1.2
    else:
        raise HTTPException(status_code=400, detail="Invalid summarization level.")

    # Generate summary
    try:
        summary_ids = model.generate(
            inputs,
            max_length=max_len,
            min_length=min_len,
            length_penalty=length_penalty,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summary generation: {str(e)}")

    return {"summary": summary}

@app.get("/")
def home():
    return {"message": "Email Summarization Model API is running."}
