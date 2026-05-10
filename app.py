from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import  T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles

#initialize our Fastapi
app = FastAPI(
    title="AI Text Summarization System", 
    description="Transformer-based text summarization using T5 and FastAPI", 
    version = '1.0'
)

# #model & tokenizer
# model_path = os.path.join(os.getcwd(), "saved_summary_model")

model = T5ForConditionalGeneration.from_pretrained(
    "./saved_summary_model",
    local_files_only=True
)

#model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    "./saved_summary_model",
    local_files_only=True)

# device
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.backends.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)
model.eval()

#templating
templates = Jinja2Templates(directory='.')


#input schema for dialogue # string
class TextInput(BaseModel):
    text:str

# Text preprocessing function
def clean_text(text):
    text = re.sub(r"\r\n", " ", text)#line
    text = re.sub(r"\s+", " ", text)#space
    text = re.sub(r"\<.*?>", " ", text)#html
    text = text.strip().lower()
    return text

# Generate summary from input text
def Generate_summary(text: str) -> str:
    text = clean_text(text)## clean

    #tokenize 
    inputs = tokenizer(
        text,
        padding = 'max_length',
        max_length = 512,
        truncation = True,
        return_tensors = 'pt'
    ).to(device)


    #generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            max_length = 150,
            num_beams = 4,## select best out of 4
            early_stopping = True
            
            
        )
    # Convert generated token IDs into readable text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens = True)
    return summary

#API end point
@app.post("/summarize/")
async def summarize(text_input: TextInput):
    try:
        summary =  Generate_summary(text_input.text)
        return {"summary":summary}
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request":request})
    