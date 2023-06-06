import cv2
import onnxruntime as rt
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from insightface.model_zoo.inswapper import INSwapper
import psutil
from enum import Enum
from insightface.app.common import Face
from insightface.utils.storage import ensure_available
import re
import subprocess

class RefacerMode(Enum):
     CPU, CUDA, COREML, TENSORRT = range(1, 5)

class Refacer:
    def __init__(self,force_cpu=False):
        self.force_cpu = force_cpu
        self.__check_encoders()
        self.__check_providers()
        self.total_mem = psutil.virtual_memory().total
        self.__init_apps()

    def __check_providers(self):
        if self.force_cpu :
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = rt.get_available_providers()
        rt.set_default_logger_severity(4)
        self.sess_options = rt.SessionOptions()
        self.sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

        if len(self.providers) == 1 and 'CPUExecutionProvider' in self.providers:
            self.mode = RefacerMode.CPU
            self.use_num_cpus = mp.cpu_count()-1
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus/3)
            print(f"CPU mode with providers {self.providers}")
        elif 'CoreMLExecutionProvider' in self.providers:
            self.mode = RefacerMode.COREML
            self.use_num_cpus = mp.cpu_count()-1
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus/3)
            print(f"CoreML mode with providers {self.providers}")
        elif 'CUDAExecutionProvider' in self.providers:
            self.mode = RefacerMode.CUDA
            self.use_num_cpus = 1
            self.sess_options.intra_op_num_threads = 1
            if 'TensorrtExecutionProvider' in self.providers:
                self.providers.remove('TensorrtExecutionProvider')
            print(f"CUDA mode with providers {self.providers}")
        """
        elif 'TensorrtExecutionProvider' in self.providers:
            self.mode = RefacerMode.TENSORRT
            #self.use_num_cpus = 1
            #self.sess_options.intra_op_num_threads = 1
            self.use_num_cpus = mp.cpu_count()-1
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus/3)
            print(f"TENSORRT mode with providers {self.providers}")
        """
        

    def __init_apps(self):
        assets_dir = ensure_available('models', 'buffalo_l', root='~/.insightface')

        model_path = os.path.join(assets_dir, 'det_10g.onnx')
        sess_face = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        self.face_detector = SCRFD(model_path,sess_face)
        self.face_detector.prepare(0,input_size=(640, 640))

        model_path = os.path.join(assets_dir , 'w600k_r50.onnx')
        sess_rec = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        self.rec_app = ArcFaceONNX(model_path,sess_rec)
        self.rec_app.prepare(0)

        model_path = 'inswapper_128.onnx'
        sess_swap = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
        self.face_swapper = INSwapper(model_path,sess_swap)

    def __prepare_faces(self, faces):
        replacements=[]
        for face in faces:
            #image1 = cv2.imread(face.origin)
            bboxes1, kpss1 = self.face_detector.autodetect(face['origin'], max_num=1)  
            if len(kpss1)<1:
                raise Exception('No face detected on "Face to replace" image')
            feat_original = self.rec_app.get(face['origin'], kpss1[0])      
            #image2 = cv2.imread(face.destination)
            _faces = self.__get_faces(face['destination'],max_num=1)
            if len(_faces)<1:
                raise Exception('No face detected on "Destination face" image')
            replacements.append((feat_original,_faces[0],face['threshold']))

        return replacements
    def __convert_video(self,video_path,output_video_path):
        if self.video_has_audio:
            print("Merging audio with the refaced video...")
            new_path = output_video_path + str(random.randint(0,999)) + "_c.mp4"
            #stream = ffmpeg.input(output_video_path)
            in1 = ffmpeg.input(output_video_path)
            in2 = ffmpeg.input(video_path)
            out = ffmpeg.output(in1.video, in2.audio, new_path,vcodec=self.ffmpeg_video_encoder)
            out.run(overwrite_output=True,quiet=True)
        else:
            new_path = output_video_path
            print("The video doesn't have audio, so post-processing is not necessary")
        
        print(f"The process has finished.\nThe refaced video can be found at {os.path.abspath(new_path)}")
        return new_path

    def __get_faces(self,frame,max_num=0):
        bboxes, kpss = self.face_detector.detect(frame,max_num=max_num,metric='default')

        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            face.embedding = self.rec_app.get(frame, kps)
            ret.append(face)
        return ret

    def __process_faces(self,frame):
        faces = self.__get_faces(frame)
        for face in faces:
            for rep_face in self.replacement_faces:
                sim = self.rec_app.compute_sim(rep_face[0], face.embedding)
                if sim>=rep_face[2]:
                    frame = self.face_swapper.get(frame, face, rep_face[1], paste_back=True)
        return frame

    def __check_video_has_audio(self,video_path):
        self.video_has_audio = False
        probe = ffmpeg.probe(video_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream is not None:
            self.video_has_audio = True
    def reface(self, video_path, faces):

        self.__check_video_has_audio(video_path)
        output_video_path = os.path.join('out',Path(video_path).name)
        self.replacement_faces=self.__prepare_faces(faces)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frames=[]
        self.k = 1
        with tqdm(total=total_frames,desc="Extracting frames") as pbar:
            while cap.isOpened():
                flag, frame = cap.read()
                if flag and len(frame)>0:
                    frames.append(frame.copy())
                    pbar.update()
                else:
                    break
            cap.release()
            pbar.close()
        
        with ThreadPoolExecutor(max_workers = self.use_num_cpus) as executor:
            results = list(tqdm(executor.map(self.__process_faces, frames), total=len(frames),desc="Processing frames"))
            for result in results:
                output.write(result)
            output.release()

        return self.__convert_video(video_path,output_video_path)
    
    def __check_encoders(self):
        self.ffmpeg_video_encoder="libx264"
        pattern = r"encoders: ([a-zA-Z0-9_]+(?: [a-zA-Z0-9_]+)*)"
        command = ['ffmpeg', '-codecs', '--list-encoders']
        commandout = subprocess.run(command, check=True, capture_output=True).stdout
        result = commandout.decode('utf-8').split('\n')
        for r in result:
            if "264" in r: 
                encoders = re.search(pattern, r).group(1).split(' ')
                #print(encoders)
                for v_c in Refacer.VIDEO_CODECS:
                    if v_c in encoders:
                        self.ffmpeg_video_encoder=v_c
                        break
        print(f"Video codec for FFMPEG: {self.ffmpeg_video_encoder}")

    VIDEO_CODECS = [
         #'h264_videotoolbox', #osx HW acceleration
         'h264_nvenc', #NVIDIA HW acceleration
         #'h264_qsv', #Intel HW acceleration
         #'h264_vaapi', #Intel HW acceleration
         #'h264_omx', #HW acceleration
         'libx264', #No HW acceleration
    ]