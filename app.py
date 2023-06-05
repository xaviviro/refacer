import gradio as gr
from refacer import Refacer

MAX_NUM_OF_FACES=8

refacer = Refacer()

n=MAX_NUM_OF_FACES

def run(*vars):
    video_path=vars[0]
    origins=vars[1:(n+1)]
    destinations=vars[(n+1):(n*2)+1]
    thresholds=vars[(n*2)+1:]

    faces = []
    for k in range(0,n):
        if origins[k] is not None and destinations[k] is not None:
            faces.append({
                'origin':origins[k],
                'destination':destinations[k],
                'threshold':thresholds[k]
            })

    return refacer.reface(video_path,faces)

origin = []
destination = []
thresholds = []

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# Refacer")
    with gr.Row():
        video=gr.Video(label="Original video")
        video2=gr.Video(label="Refaced video",interactive=False)

    for i in range(0,MAX_NUM_OF_FACES):
        with gr.Tab(f"Face #{i+1}"):
            with gr.Row():
                origin.append(gr.Image(label="Face to replace"))
                destination.append(gr.Image(label="Destination face"))
            with gr.Row():
                thresholds.append(gr.Slider(label="Threshold",minimum=0.0,maximum=1.0,value=0.2))
    with gr.Row():
        button=gr.Button("Reface", variant="primary")

    button.click(fn=run,inputs=[video]+origin+destination+thresholds,outputs=[video2])

#demo.launch(share=True,server_name="0.0.0.0", show_error=True)
demo.queue().launch(show_error=True,share=True)