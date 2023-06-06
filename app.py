import gradio as gr
from refacer import Refacer
import argparse

parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", help="Max number of faces on UI", default=5)
parser.add_argument("--force_cpu", help="Force CPU mode", default=False,action="store_true")
parser.add_argument("--share_gradio", help="Share Gradio", default=False,action="store_true")
args = parser.parse_args()

refacer = Refacer(force_cpu=args.force_cpu)

num_faces=args.max_num_faces

def run(*vars):
    video_path=vars[0]
    origins=vars[1:(num_faces+1)]
    destinations=vars[(num_faces+1):(num_faces*2)+1]
    thresholds=vars[(num_faces*2)+1:]

    faces = []
    for k in range(0,num_faces):
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

    for i in range(0,num_faces):
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
demo.queue().launch(show_error=True,share=args.share_gradio)