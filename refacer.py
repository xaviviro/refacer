import cv2
import insightface
import onnxruntime
import sys
from insightface.app import FaceAnalysis
sys.path.insert(1, './recognition')
from scrfd import SCRFD
from arcface_onnx import ArcFaceONNX
import os.path as osp
import os
from pathlib import Path
from tqdm import tqdm
import ffmpeg
import random

class Refacer:

    def __init__(self):
        onnxruntime.set_default_logger_severity(0)

        self.face_app = FaceAnalysis(name='buffalo_l')
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        assets_dir = osp.expanduser('~/.insightface/models/buffalo_l')

        self.face_detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
        self.face_detector.prepare(0)

        model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
        self.rec_app = ArcFaceONNX(model_path)
        self.rec_app.prepare(0)

        self.face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True, providers=['CoreMLExecutionProvider','CUDAExecutionProvider'])

    def __prepare_faces(self, faces):
        replacements=[]
        for face in faces:
            #image1 = cv2.imread(face.origin)
            bboxes1, kpss1 = self.face_detector.autodetect(face['origin'], max_num=1)      
            feat_original = self.rec_app.get(face['origin'], kpss1[0])      
            #image2 = cv2.imread(face.destination)
            _faces = self.face_app.get(face['destination'],max_num=1)
            replacements.append((feat_original,_faces[0],face['threshold']))

        return replacements
    def __convert_video(self,video_path,output_video_path):
        new_path = output_video_path + str(random.randint(0,999)) + "_c.mp4"
        #stream = ffmpeg.input(output_video_path)
        in1 = ffmpeg.input(output_video_path)
        in2 = ffmpeg.input(video_path)
        out = ffmpeg.output(in1.video, in2.audio, new_path,vcodec="libx264")
        out.run()
        return new_path

    def reface(self, video_path, faces):        
        output_video_path = os.path.join('out',Path(video_path).name)
        replacement_faces=self.__prepare_faces(faces)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        #probe = ffmpeg.probe(video_path)
        #video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        #print(video_stream)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                flag, frame = cap.read()
                if flag and len(frame)>0:
                    #pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    faces = self.face_app.get(frame)
                    res = frame.copy()

                    for face in faces:
                        for rep_face in replacement_faces:
                            sim = self.rec_app.compute_sim(rep_face[0], face.embedding)
                            if sim>=rep_face[2]:
                                res = self.face_swapper.get(res, face, rep_face[1], paste_back=True)

                    output.write(res)
                    pbar.update(1)
                else:
                    break
        
        pbar.close()
        cap.release()
        output.release()

        return self.__convert_video(video_path,output_video_path)
    
        