"""Model operations for text and audio generation."""
from typing import Literal

import torch
from transformers import AutoModel, AutoProcessor, pipeline

import utils


class TextModel:
    """Text generation model using TinyLlama."""
    def __init__(self):
        """Initialize the text model and set device."""
        self.pipe = None
        self.device_name = "mps" if torch.backends.mps.is_available() else "cpu"

    def load_pipeline(self):
        """Load the text generation pipeline."""
        self.pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,
            device=self.device_name,
        )
        print("TextModel is loaded.")

    def predict(self, user_message):
        """Generate a response to the user message in pirate style."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly chatbot who always responds in the style of a pirate"
                ),
            },
            {"role": "user", "content": user_message},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )
        outputs_data = outputs[0]["generated_text"].split("<|assistant|>")
        if len(outputs_data) != 2:
            return "unexpected output len"
        result = outputs_data[1].strip()
        return result


class AudioModel:
    """Audio generation model using Bark."""
    VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"]

    def __init__(self):
        """Initialize the audio model with default preset."""
        self.processor = None
        self.model = None

    def load_audio_model(self) -> tuple:
        """Load the audio processor and model."""
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = AutoModel.from_pretrained("suno/bark-small")
        print("AudioModel is loaded.")

    def generate_audio(self,preset:str, prompt: str):
        """Generate audio from a text prompt."""
        if self.processor is None or self.model is None:
            return None, None
        inputs = self.processor(
            text=[prompt], return_tensors="pt", voice_preset=preset
        )
        output = self.model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()
        sample_rate = self.model.generation_config.sample_rate
        audio_buffer = utils.audio_array_to_buffer(output, sample_rate)
        return audio_buffer, sample_rate
