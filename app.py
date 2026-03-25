import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Age Detection using Machine Learning")

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (277, 277), [104,117,123], swapRB=False)

    faceNet.setInput(blob)
    detection = faceNet.forward()

    bboxs = []

    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]

        if confidence > 0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)

            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    return frame,bboxs


uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    frame = np.array(image)

    frame,bboxs = faceBox(faceNet,frame)

    for bbox in bboxs:

        face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]

        blob = cv2.dnn.blobFromImage(face,1.0,(227,227),MODEL_MEAN_VALUES,swapRB=False)

        ageNet.setInput(blob)
        agePred = ageNet.forward()

        age = ageList[agePred[0].argmax()]

        label = f"Age: {age}"

        cv2.putText(frame,label,(bbox[0],bbox[1]-10),
        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)

    st.image(frame, channels="BGR")
