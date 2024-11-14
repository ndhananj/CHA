import os
import gradio as gr
from transformers import pipeline
import numpy as np

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'OPENAI API KEY NOT FOUND')

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    return transcriber({"sampling_rate": sr, "raw": y})["text"]  

demo = gr.Interface(
    transcribe,
    gr.Audio(sources="microphone"),
    "text",
)

demo.launch()