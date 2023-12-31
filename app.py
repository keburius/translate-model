from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from model import TranslationModel
from fastapi.middleware.cors import CORSMiddleware  # New import

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# Initialize the TranslationModel class and keep it in memory.
translation_model = TranslationModel()


class TranslateType(BaseModel):
    text: str
    target_lang: str


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/translate")
async def home(request: TranslateType):
    input_text = request.text
    output_language = request.target_lang

    if input_text and output_language:
        predicted_translation = translation_model.translate_text(input_text, output_language)
        return {'translation': predicted_translation, 'confidence': 1}

    return {'error': 'Invalid request or missing data'}
