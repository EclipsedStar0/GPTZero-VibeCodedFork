"""
This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
uvicorn_access = logging.getLogger("uvicorn.access")
uvicorn_access.disabled = True


from contextlib import asynccontextmanager
from typing import Optional, Tuple, Any
from fastapi import FastAPI, Form, Request, HTTPException
import gradio as gr
import uvicorn
from database import DB
from HTML_MD_Components import noticeBoardMarkDown, bannerHTML, emailHTML, discordHTML
from config import settings
from model import GPT2PPL

logger = logging.getLogger(__name__)

CUSTOM_PATH: str = "/"

model: Optional[GPT2PPL] = None
database: Optional[DB] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, database
    if model is None:
        model = GPT2PPL()
    if database is None:
        database = DB()
    yield
    model = None
    database = None

app: FastAPI = FastAPI(lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.debug("Response: %s %s - Status: %d", request.method, request.url.path, response.status_code)
    return response

@app.post("/postdb")
def uploadDataBase(email: str = Form(), request: Request = None) -> str:
    ip = request.client.host if request and request.client else "unknown"
    logger.info("Database upload request from IP: %s", ip)
    try:
        database.set(ip, email)
        return "Email Sent"
    except Exception as e:
        logger.error("Database upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Failed to save email.")

def inference(sentence: str) -> Any:
    logger.info("Inference request received, text length: %d", len(sentence))
    try:
        if not sentence or not sentence.strip():
            return {"error": "Input text is empty or invalid."}, "Please provide valid text input."
        return model(sentence=sentence)
    except Exception as e:
        logger.error("Inference error: %s", str(e))
        return {"error": "An error occurred during analysis.", "details": str(e)}, "Error: Unable to process the request."

@app.get("/infer")
def infer(sentence: Optional[str] = None) -> Any:
    if not sentence:
        raise HTTPException(status_code=400, detail="Missing required parameter: sentence")
    if not sentence.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    logger.info("API inference request received, text length: %d", len(sentence))
    try:
        return model(sentence=sentence)
    except Exception as e:
        logger.error("API inference error: %s", str(e))
        raise HTTPException(status_code=500, detail="An error occurred during analysis.")

with gr.Blocks(title="SG-GPTZero") as io:
    with gr.Row():
         gr.HTML(bannerHTML, visible=True)
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=8):
            gr.Markdown('<h1 style="text-align: center;">SG-GPTZero <a style="text-decoration:none" href="https://github.com/BurhanUlTayyab/GPTZero">(Code)</a></h1>')
        with gr.Column(scale=1, elem_id="discord"):
            gr.HTML(discordHTML, visible=True)
    with gr.Row():
        gr.Markdown("Use SG-GPTZero to determine if the text is written by AI or Human.")
    with gr.Row(elem_id="row1"):
        with gr.Column(scale=10):
            InputTextBox: gr.Textbox = gr.Textbox(lines=7, placeholder="Please Insert your text(s) here", label="Texts")
            sumbit_btn: gr.Button = gr.Button("Submit", elem_id="submit")
        with gr.Column(scale=10):
            OutputLabels: gr.JSON = gr.JSON(label="Output")
            OutputTextBox: gr.Textbox = gr.Textbox(show_label=False)
        sumbit_btn.click(inference, inputs=InputTextBox, outputs=[OutputLabels, OutputTextBox], api_name="infer")
    with gr.Row():
        with gr.Column():
            gr.Markdown(noticeBoardMarkDown(), visible=True)
    with gr.Row():
        gr.Markdown('# <span style="color:#006400">Register</span> here for updates.')
    with gr.Row():
        with gr.Column(scale=5):
            emailTextBox: gr.HTML = gr.HTML(emailHTML)
        with gr.Column(scale=5):
            pass
 
    with gr.Row():
        gr.Markdown('<span style="color:red">Do you want to train Computer vision models faster and with less data?. Visit [Ailiverse](https://ailiverse.com)</span> <br> <span style="color:gray"><p>Powered by Ailiverse </p></span>', elem_id="advertisment")
    with gr.Row():
        gr.Markdown('For <a style="text-decoration:none;color:gray" href="mailto:gptzero@ailiverse.com" target="_blank">feedback</a>, contact us at gptzero@ailiverse.com', elem_id="code_feedback")

app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH)

if __name__ == "__main__":
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.reload, forwarded_allow_ips="*", proxy_headers=True)