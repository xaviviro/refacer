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
import numpy as np
from esrgan_onnx import ESRGAN
from gfpgan_onnx import GFPGAN

class RefacerMode(Enum):
     CPU, CUDA, COREML, TENSORRT = range(1, 5)

class Refacer:
    def __init__(self,force_cpu=False,colab_performance=False):
        self.first_face = False
        self.force_cpu = force_cpu
        self.colab_performance = colab_performance
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
        elif self.colab_performance:
            self.mode = RefacerMode.TENSORRT
            self.use_num_cpus = mp.cpu_count()-1
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus/3)
            print(f"TENSORRT mode with providers {self.providers}")
        elif 'CoreMLExecutionProvider' in self.providers:
            self.mode = RefacerMode.COREML
            self.use_num_cpus = mp.cpu_count()-1
            self.sess_options.intra_op_num_threads = int(self.use_num_cpus/3)
            print(f"CoreML mode with providers {self.providers}")
        elif 'CUDAExecutionProvider' in self.providers:
            self.mode = RefacerMode.CUDA
            self.use_num_cpus = 2
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
        self.face_swapper_input_size = self.face_swapper.input_size[0]
        #print("INSwapper resolution = ",self.face_swapper_input_size)



    def prepare_faces(self, faces):
        self.replacement_faces=[]
        for face in faces:
            #image1 = cv2.imread(face.origin)
            if "origin" in face:
                face_threshold = face['threshold']
                bboxes1, kpss1 = self.face_detector.autodetect(face['origin'], max_num=1)  
                if len(kpss1)<1:
                    raise Exception('No face detected on "Face to replace" image')
                feat_original = self.rec_app.get(face['origin'], kpss1[0])
            else:
                face_threshold = 0
                self.first_face = True
                feat_original = None
                print('No origin image: First face change')
            #image2 = cv2.imread(face.destination)
            _faces = self.__get_faces(face['destination'],max_num=1)
            if len(_faces)<1:
                raise Exception('No face detected on "Destination face" image')
            self.replacement_faces.append((feat_original,_faces[0],face_threshold))

    def __convert_video(self,video_path,output_video_path):
        if self.video_has_audio:
            print("Merging audio with the refaced video...")
            new_path = output_video_path + str(random.randint(0,999)) + "_c.mp4"
            #stream = ffmpeg.input(output_video_path)
            in1 = ffmpeg.input(output_video_path)
            in2 = ffmpeg.input(video_path)
            out = ffmpeg.output(in1.video, in2.audio, new_path,video_bitrate=self.ffmpeg_video_bitrate,vcodec=self.ffmpeg_video_encoder)
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

    def paste_upscale(self, bgr_fake, M, img):
        upsk_face, self.scale_factor = self.face_upscaler_model.get(bgr_fake)
        M_scale = M * self.scale_factor
        target_img = img
        IM = cv2.invertAffineTransform(M_scale)

        face_matte = np.full((target_img.shape[0],target_img.shape[1]), 255, dtype=np.uint8)
        
        ##Generate white square sized as a upsk_face
        img_matte = np.full((upsk_face.shape[0],upsk_face.shape[1]), 255, dtype=np.uint8) 
        ##Transform white square back to target_img
        img_matte = cv2.warpAffine(img_matte, IM, (target_img.shape[1], target_img.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0.0) 
        ##Blacken the edges of face_matte by 1 pixels (so the mask in not expanded on the image edges)
        img_matte[:1,:] = img_matte[-1:,:] = img_matte[:,:1] = img_matte[:,-1:] = 0 
        #Detect the affine transformed white area
        mask_h_inds, mask_w_inds = np.where(img_matte==255) 
        #Calculate the size (and diagonal size) of transformed white area width and height boundaries
        mask_h = np.max(mask_h_inds) - np.min(mask_h_inds) 
        mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
        mask_size = int(np.sqrt(mask_h*mask_w))
        #Calculate the kernel size for eroding img_matte by kernel (insightface empirical guess for best size was max(mask_size//10,10))
        k = max(mask_size//12, 8)
        kernel = np.ones((k,k),np.uint8)
        img_matte = cv2.erode(img_matte,kernel,iterations = 1)
        #Calculate the kernel size for blurring img_matte by blur_size (insightface empirical guess for best size was max(mask_size//20, 5))
        k = max(mask_size//24, 4) 
        kernel_size = (k, k)
        blur_size = tuple(2*i+1 for i in kernel_size)
        img_matte = cv2.GaussianBlur(img_matte, blur_size, 0)
        
        #Normalize images to float values and reshape
        img_matte = img_matte.astype(np.float32)/255
        face_matte = face_matte.astype(np.float32)/255
        img_matte = np.minimum(face_matte, img_matte)
        img_matte = np.reshape(img_matte, [img_matte.shape[0],img_matte.shape[1],1]) 
        ##Transform upcaled face back to target_img
        paste_face = cv2.warpAffine(upsk_face, IM, (target_img.shape[1], target_img.shape[0]), borderMode=cv2.BORDER_REPLICATE) 
        ##Re-assemble image
        paste_face = img_matte * paste_face
        paste_face = paste_face + (1-img_matte) * target_img.astype(np.float32) 
        return paste_face.astype(np.uint8)

    def process_first_face(self,frame):
        faces = self.__get_faces(frame,max_num=1)
        if len(faces) != 0:
            if not self.upscale_en: 
                #print('\nRun native paste_back')
                frame = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=True)
            else: 
                #print('\nRun upscale')
                bgr_fake, M = self.face_swapper.get(frame, face, self.replacement_faces[0][1], paste_back=False)
                frame = self.paste_upscale(bgr_fake,M,frame)
        return frame

    def process_faces(self,frame):
        faces = self.__get_faces(frame,max_num=0)
        for rep_face in self.replacement_faces:
            for i in range(len(faces) - 1, -1, -1):
                sim = self.rec_app.compute_sim(rep_face[0], faces[i].embedding)
                if sim>=rep_face[2]:
                    if not self.upscale_en: 
                        #print('\nRun native paste_back')
                        frame = self.face_swapper.get(frame, face, rep_face[1], paste_back=True)
                    else: 
                        #print('\nRun upscale')
                        bgr_fake, M = self.face_swapper.get(frame, face, rep_face[1], paste_back=False)
                        frame = self.paste_upscale(bgr_fake,M,frame)
                    del faces[i]
                    break
        return frame

    def __check_video_has_audio(self,video_path):
        self.video_has_audio = False
        probe = ffmpeg.probe(video_path)
        audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        if audio_stream is not None:
            self.video_has_audio = True
        
    def reface_group(self, faces, frames, output):
        with ThreadPoolExecutor(max_workers = self.use_num_cpus) as executor:
            if self.first_face:
                results = list(tqdm(executor.map(self.process_first_face, frames), total=len(frames),desc="Processing frames"))
            else:
                results = list(tqdm(executor.map(self.process_faces, frames), total=len(frames),desc="Processing frames"))
            for result in results:
                output.write(result)

    def reface(self, video_path, faces, upscaler):
        self.upscale_en = False
        if upscaler != 'None': 
            self.upscale_en = True
            model_path = osp.join('upscaler_models',upscaler)
            sess_upsk = rt.InferenceSession(model_path, self.sess_options, providers=self.providers)
            if 'GFPGAN' in str(upscaler):
                self.face_upscaler_model = GFPGAN(sess_upsk)
                #print('\nGFPGAN upscaling.')
            else:
                self.face_upscaler_model = ESRGAN(sess_upsk)
                #print('\nESRGAN upscaling.')        
        #else: print('\nNot upscaling.')     
        self.__check_video_has_audio(video_path)
        output_video_path = os.path.join('out',Path(video_path).name)
        self.prepare_faces(faces)

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
                if (len(frames) > 1000):
                    self.reface_group(faces,frames,output)
                    frames=[]

            cap.release()
            pbar.close()

        self.reface_group(faces,frames,output)
        frames=[]
        output.release()
        
        return self.__convert_video(video_path,output_video_path)
    
    def __try_ffmpeg_encoder(self, vcodec):
        print(f"Trying FFMPEG {vcodec} encoder")
        command = ['ffmpeg', '-y', '-f','lavfi','-i','testsrc=duration=1:size=1280x720:rate=30','-vcodec',vcodec,'testsrc.mp4']
        try:
            subprocess.run(command, check=True, capture_output=True).stderr
        except subprocess.CalledProcessError as e:
            print(f"FFMPEG {vcodec} encoder doesn't work -> Disabled.")
            return False
        print(f"FFMPEG {vcodec} encoder works")
        return True
        
    def __check_encoders(self):
        self.ffmpeg_video_encoder='libx264'
        self.ffmpeg_video_bitrate='0'

        pattern = r"encoders: ([a-zA-Z0-9_]+(?: [a-zA-Z0-9_]+)*)"
        command = ['ffmpeg', '-codecs', '--list-encoders']
        commandout = subprocess.run(command, check=True, capture_output=True).stdout
        result = commandout.decode('utf-8').split('\n')
        for r in result:
            if "264" in r: 
                encoders = re.search(pattern, r).group(1).split(' ')
                for v_c in Refacer.VIDEO_CODECS:
                    for v_k in encoders:
                        if v_c == v_k:
                            if self.__try_ffmpeg_encoder(v_k):
                                self.ffmpeg_video_encoder=v_k
                                self.ffmpeg_video_bitrate=Refacer.VIDEO_CODECS[v_k]
                                print(f"Video codec for FFMPEG: {self.ffmpeg_video_encoder}")
                                return

    VIDEO_CODECS = {
         'h264_videotoolbox':'0', #osx HW acceleration
         'h264_nvenc':'0', #NVIDIA HW acceleration
         #'h264_qsv', #Intel HW acceleration
         #'h264_vaapi', #Intel HW acceleration
         #'h264_omx', #HW acceleration
         'libx264':'0' #No HW acceleration
    }
