"""Main FastAPI application for text and audio generation APIs."""
import asyncio
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Body, FastAPI, Query, status
from fastapi.responses import StreamingResponse

from model_work import AudioModel, TextModel
from schemas import StudentRequestModel, TextRequestModel, TextResponseModel

ml_models = {}

@asynccontextmanager
async def liefspan(app:FastAPI):
    """Lifespan event handler to load and unload ML models."""
    text_m_obj = TextModel()
    text_m_obj.load_pipeline()
    ml_models["text_m_obj"] = text_m_obj

    audio_m_obj = AudioModel()
    audio_m_obj.load_audio_model()
    ml_models["audio_m_obj"] = audio_m_obj

    yield
    ml_models.clear()

app = FastAPI(lifespan=liefspan)

@app.get("/")
def home():
    """Health check endpoint."""
    return {"message": "Hello World"}

@app.post("/get_student")
def get_student(body: StudentRequestModel = Body(...)) -> TextResponseModel:
    """Get student info and return a response."""
    start_time = time.time()
    print('body', body)
    result = "OK"
    return TextResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )

@app.post("/sync")
def sync_prediction(prompt: str) -> TextResponseModel:
    """Synchronous prediction endpoint (simulated delay)."""
    start_time = time.time()
    time.sleep(5)
    result = "OK"
    print('prompt', prompt)
    return TextResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )

@app.post("/async")
async def async_prediction() -> TextResponseModel:
    """Asynchronous prediction endpoint (simulated delay)."""
    start_time = time.time()
    await asyncio.sleep(5)
    result = "OK"
    return TextResponseModel(
        execution_time=int(time.time() - start_time), result=result
    )

@app.post("/text_gen")
def serve_text_gen(body: TextRequestModel = Body(...)) -> TextResponseModel:
    """Generate text using the loaded text model."""
    start_time = time.time()
    generated_text = ml_models["text_m_obj"].predict(user_message=body.prompt)
    return TextResponseModel(
        execution_time=int(time.time() - start_time), result=generated_text
    )

@app.get(
    "/audio_gen",
    responses={status.HTTP_200_OK: {"content": {"audio/wav": {}}}},
    response_class=StreamingResponse,
)
async def serve_audio_gen(
    prompt: str = Query(...), preset: AudioModel.VoicePresets = Query(default="v2/en_speaker_9")
) -> StreamingResponse:
    """Generate audio from text using the loaded audio model."""
    audio_buffer, _ = ml_models["audio_m_obj"].generate_audio(preset,prompt)
    return StreamingResponse(
        audio_buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=generated_audio.wav"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
