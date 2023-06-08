import gradio as gr
from refacer import Refacer
import argparse
import ngrok

parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, help="Max number of faces on UI", default=5)
parser.add_argument("--force_cpu", help="Force CPU mode", default=False, action="store_true")
parser.add_argument("--share_gradio", help="Share Gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, help="Server IP address", default="127.0.0.1")
parser.add_argument("--server_port", type=int, help="Server port", default=7860)
parser.add_argument("--colab_performance", help="Use in colab for better performance", default=False,action="store_true")
parser.add_argument("--ngrok", type=str, help="Use ngrok", default=None)
parser.add_argument("--ngrok_region", type=str, help="ngrok region", default="us")
args = parser.parse_args()

refacer = Refacer(force_cpu=args.force_cpu,colab_performance=args.colab_performance)

num_faces=args.max_num_faces

# Connect to ngrok for ingress
def connect(token, port, options):
    account = None
    if token is None:
        token = 'None'
    else:
        if ':' in token:
            # token = authtoken:username:password
            token, username, password = token.split(':', 2)
            account = f"{username}:{password}"

    # For all options see: https://github.com/ngrok/ngrok-py/blob/main/examples/ngrok-connect-full.py
    if not options.get('authtoken_from_env'):
        options['authtoken'] = token
    if account:
        options['basic_auth'] = account


    try:
        public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
    except Exception as e:
        print(f'Invalid ngrok authtoken? ngrok connection aborted due to: {e}\n'
              f'Your token: {token}, get the right one on https://dashboard.ngrok.com/get-started/your-authtoken')
    else:
        print(f'ngrok connected to localhost:{port}! URL: {public_url}\n'
               'You can use this link after the launch is complete.')


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
        video=gr.Video(label="Original video",format="mp4")
        video2=gr.Video(label="Refaced video",interactive=False,format="mp4")

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
    
if args.ngrok is not None:
    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})
    
#demo.launch(share=True,server_name="0.0.0.0", show_error=True)
demo.queue().launch(show_error=True,share=args.share_gradio,server_name=args.server_name,server_port=args.server_port)
