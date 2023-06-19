from refacer import Refacer
from os.path import exists
import argparse
import cv2

parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--force_cpu", help="Force CPU mode", default=False, action="store_true")
parser.add_argument("--colab_performance", help="Use in colab for better performance", default=False,action="store_true")
parser.add_argument("--face", help="Face to replace (ex: <src>,<dst>,<thresh=0.2>)", nargs='+', action="append", required=True)
parser.add_argument("--video", help="Video to parse", required=True)
args = parser.parse_args()

refacer = Refacer(force_cpu=args.force_cpu,colab_performance=args.colab_performance)

def run(video_path,faces):
    video_path_exists = exists(video_path)
    if video_path_exists == False:
        print ("Can't find " + video_path)
        return

    faces_out = []
    for face in faces:
        face_str = face[0].split(",")
        origin = exists(face_str[0])
        if origin == False:
            print ("Can't find " + face_str[0])
            return
        destination = exists(face_str[1])
        if destination == False:
            print ("Can't find " + face_str[1])
            return
        
        faces_out.append({
                'origin':cv2.imread(face_str[0]),
                'destination':cv2.imread(face_str[1]),
                'threshold':float(face_str[2])
            })

    return refacer.reface(video_path,faces_out)

run(args.video, args.face)
