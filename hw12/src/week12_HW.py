import cv2
import numpy as np
import time
import argparse
from multiprocessing import Process, Queue
import threading

import os
import os.path as osp
import sys
BUILD_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "build/service/")
sys.path.insert(0, BUILD_DIR)

import grpc
from concurrent import futures
import fib_pb2
import fib_pb2_grpc

import mediapipe as mp
mp_hands = mp.solutions.hands
mp_object_detection = mp.solutions.object_detection
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

class FibCalculatorServicer(fib_pb2_grpc.FibCalculatorServicer):

    def __init__(self, q_mode):
        self.q_mode = q_mode
        pass
    
    def Compute(self, request, cotext):
        
        mode = request.order
        self.q_mode.put(mode)
        response = fib_pb2.FibResponse()
        response.value = mode
        print(f'mode change to {mode}')
        return response

def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
    pipeline = (
        "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)1920, height=(int)1080, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue ! "
        "nvvidconv flip-method=2 ! "
            "video/x-raw, "
            "width=(int)1920, height=(int)1080, "
            "format=(string)BGRx, framerate=(fraction)30/1 ! "
        "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
        "appsink"
    )
    # Complete the function body
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (480, 270))
            # print(time.strftime('%X'), frame.shape)
            queue.put(frame)
    except KeyboardInterrupt as e:
        cap.release()


def gstreamer_rtmpstream(queue, mode):
    # Use the provided pipeline to construct the video writer in opencv

    pipeline = (
        "appsrc ! "
            "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
            "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "flvmux ! "
        'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    )
    # Complete the function body
    # You can apply some simple computer vision algorithm here
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(pipeline, 0, 30.0, (480, 270))
    
    sample_rate = 5
    i_num = 0
    last_result = []
    last_result_pose = None
    last_result_obj = []
    last_mode = 0
    try:
        hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        objects = mp_object_detection.ObjectDetection(min_detection_confidence=0.1)
        poses = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True, min_detection_confidence=0.5)
        while True:
            if queue.empty():
                # print("queue is empty")
                continue
            i_num += 1
            frame = queue.get()
            # last_mode = 1
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # print(frame.shape)
            # frame = cv2.Canny(frame, 30, 150)
            # frame = np.stack([frame, frame, frame], axis=-1)
            # frame[:, :, 1] = 0
            # out.write(frame)
            # last_time = time.time()
            if i_num % sample_rate == 0:
                if last_mode == 1:
                    # for hand_landmarks in last_result:
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:
                        last_result = results.multi_hand_landmarks
                    else:
                        last_result = []
                elif last_mode == 2:
                    results_obj = objects.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results_obj.detections:
                        last_result_obj = results_obj.detections
                    else:
                        last_result_obj = []
                    # for detection in last_result_obj:
                elif last_mode == 3:
                    results_pose = poses.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if results_pose.pose_landmarks:
                        last_result_pose = results_pose.pose_landmarks
                    else:
                        last_result_pose = None
                else:
                    pass
                    
            if last_mode == 1:
                for hand_landmarks in last_result:
                    mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())  
            if last_mode == 2:
                for detection in last_result_obj:
                    mp_drawing.draw_detection(frame, detection)
            elif last_mode == 3:
                if last_result_pose:
                            mp_drawing.draw_landmarks(
                                frame,
                                last_result_pose,
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            else:
                pass
            print("rtmp running...")
            print(f'mode: {last_mode}')
            out.write(frame)
            if mode.empty():
                continue
            last_mode = mode.get()
            print(f'mode change to {last_mode}')

            if cv2.waitKey(1) == ord('q'):
                break
        out.release()
        cv2.destroyAllWindows()
    except:
        out.release()
        cv2.destroyAllWindows()

def get_mode(mode):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FibCalculatorServicer(mode)
    fib_pb2_grpc.add_FibCalculatorServicer_to_server(servicer, server)

    try:
        server.add_insecure_port('0.0.0.0:8080')
        server.start()
        print('Run gRPC server at 0.0.0.0:8080')
        print(servicer.q_mode)
        server.wait_for_termination()
    except KeyboardInterrupt:
        pass

# Complelte the code
if __name__ == '__main__':
    queue = Queue(maxsize=10)
    mode = Queue(maxsize=1)

    p = Process(target = gstreamer_camera, args=[queue])
    q = Process(target = get_mode, args=[mode])
    main = Process(target = gstreamer_rtmpstream, args=[queue, mode])
    p.start()
    main.start()
    q.start()
    # gstreamer_rtmpstream(queue, mode)
    # p.terminate()
    # p.join()
    
