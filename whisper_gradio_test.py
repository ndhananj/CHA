import gradio as gr
import time
import numpy as np
from transformers import pipeline

def sequential_process(audio):
    if audio is None:
        return ["No audio recorded", "", "", "", ""]
    
    # First pane shows audio recording status
    status = "Audio recorded successfully"
    yield [status, "", "", "", ""]
    
    # Second pane: Whisper transcription
    time.sleep(0.5)  # Give UI time to update
    sr, y = audio
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    
    transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")
    transcribed_text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    yield [status, f"Transcription: {transcribed_text}", "", "", ""]
    
    # Third pane: Word count and statistics
    time.sleep(1)
    words = transcribed_text.split()
    stats = f"Words: {len(words)}, Characters: {len(transcribed_text)}"
    yield [status, f"Transcription: {transcribed_text}", stats, "", ""]
    
    # Fourth pane: Timestamp and metadata
    time.sleep(1)
    metadata = f"Processed at {time.strftime('%H:%M:%S')}, Audio duration: {len(y)/sr:.2f} seconds"
    yield [status, f"Transcription: {transcribed_text}", stats, metadata, ""]
    
    # Fifth pane: Summary
    time.sleep(1)
    summary = "Processing complete! Audio has been transcribed and analyzed."
    yield [status, f"Transcription: {transcribed_text}", stats, metadata, summary]

# Create the interface
with gr.Blocks() as demo:
    gr.Markdown("# Audio Processing Pipeline")
    
    # Input audio component
    input_audio = gr.Audio(
        sources="microphone",
        type="numpy",
        label="Record Audio"
    )
    
    # Button to start processing
    process_btn = gr.Button("Process Recording")
    
    # Output panes
    output1 = gr.Textbox(label="Recording Status")
    output2 = gr.Textbox(label="Transcription")
    output3 = gr.Textbox(label="Statistics")
    output4 = gr.Textbox(label="Metadata")
    output5 = gr.Textbox(label="Summary")
    
    # Connect the button to the function
    process_btn.click(
        fn=sequential_process,
        inputs=[input_audio],
        outputs=[output1, output2, output3, output4, output5]
    )

# Launch the interface
demo.queue().launch()