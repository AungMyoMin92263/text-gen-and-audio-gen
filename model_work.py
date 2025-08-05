
from typing import Literal
import numpy as np
from io import BytesIO
from transformers import AutoProcessor, AutoModel, Pipeline, pipeline
from typing import Literal
import torch
import utils


class textModel():
    def __init__(self):
        self.pipeline = None
        self.device_name = None
        if torch.backends.mps.is_available():
            self.device_name = "mps" #cuda
        else:
            self.device_name = "cpu"



    def load_pipeline(self):
        self.pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device=self.device_name)
        print("textModel is loaded ")


    def predict(self,user_message):
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot who always responds in the style of a pirate",
            },
            {"role": "user", "content": user_message},
        ]
        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        outputs_data = outputs[0]["generated_text"].split("<|assistant|>")
        if len(outputs_data) != 2:
            return "unexpected output len"

        result = outputs_data[1].strip()
        return result





class audioModel():
    VoicePresets = Literal["v2/en_speaker_1", "v2/en_speaker_9"]


    def __init__(self):
        self.preset = "v2/en_speaker_9"
        self.processor = None
        self.model = None

    def load_audio_model(self) -> tuple[AutoProcessor, AutoModel]:

        #Download the small bark processor which prepares input text prompt for the core model
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")


        #Download the bark model which will be used to generate the output audio.
        self.model = AutoModel.from_pretrained("suno/bark-small")

        print("audioModel is loaded ")
    

    def generate_audio(
        self,
        prompt: str ) -> tuple[np.array, int]:

        if self.processor is None or self.model is None:
            return
        

        # Preprocess text prompt with a speaker voice preset embedding and return a Pytorch tensor array of tokenized inputs using return_tensors="pt"
        inputs = self.processor(text=[prompt], return_tensors="pt", voice_preset=self.preset)


        # Generate an audio array that contains amplitude values of the synthesized audio signal over time.
        output = self.model.generate(**inputs, do_sample=True).cpu().numpy().squeeze()

        # Obtain the sampling rate from model generating configurations which can be used to produce the audio.
        sample_rate = self.model.generation_config.sample_rate
        audio_buffer = utils.audio_array_to_buffer(output,sample_rate)


        return audio_buffer