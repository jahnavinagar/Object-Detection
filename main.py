import cv2
import numpy as np


thres = 0.45  
nms_threshold = 0.3  


cap = cv2.VideoCapture(0)  
cap.set(3, 1280)  
cap.set(4, 720)  
cap.set(10, 70)   


classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")


configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


save_output = True
if save_output:
    output_file = 'output_detection.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (1280, 720))


while True:
    success, img = cap.read()
    if not success:
        break

    
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    if len(bbox) > 0:
        
        bbox = list(bbox)
        confs = list(map(float, np.array(confs).reshape(-1)))  
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)


        if len(indices) > 0:
            for i in indices.flatten():  
                box = bbox[i]  
                classId = classIds[i]  
                confidence = confs[i]  
                print(classId,box)

                
                cv2.rectangle(img, box, color=(0, 0, 0), thickness=2)

                
                cv2.putText(img, f'{classNames[classId - 1].upper()} {int(confidence * 100)}%',
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    
    detected_count = len(indices) if len(bbox) > 0 else 0
    cv2.putText(img, f'Total Objects Detected: {detected_count}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    cv2.imshow("Object Detection Output", img)

    
    if save_output:
        out.write(img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
