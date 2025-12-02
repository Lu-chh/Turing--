import numpy as np
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from matplotlib.font_manager import FontProperties
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = r"pose.mp4"
cap = cv2.VideoCapture(video_path)

#得到图像尺寸和视频帧率
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("FPS:", fps)
print("宽度:", width)
print("高度:", height)

#设定处理后视频格式
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("new_pose.mp4", fourcc, fps, (width, height))

#预设定repl.csv
csv_file = open("repl.csv", "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "index", "x", "y", "z", "visibility"])


total_v = 0    #每一帧速度的和
frame_idx = 0  #视频帧编号
last = None    #上一帧的关键点

#处理每一帧
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        #获取关键点
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        #绘制关键点
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        #获取关键点集
        landmarks = results.pose_landmarks.landmark

        #将关键点位置写入repl.csv
        for i, lm in enumerate(landmarks):
            csv_writer.writerow([frame_idx, i, lm.x, lm.y, lm.z, lm.visibility])

        #判断是否是第0帧
        if last == None:
            last = landmarks[25]
        else:
            lk = landmarks[25] #左膝位置

            ds = (((lk.x - last.x) ** 2 + (lk.y - last.y) ** 2) ** 0.5) * 3.7 #一帧的位移
            v = ds * fps * 2                                                  #计算速度
            total_v += v                                                      #累加计入速度和 用于求平均速度

            cv2.putText(image,                                                #在图中标注每一帧速度
                        f"Speed: {v:.2f} m/s",
                        (1000, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)
            last = landmarks[25]                                              #将本节点记为last

        #写入new_pose.mp4
        out.write(image)
        #展示处理后的视频
        cv2.imshow("MediaPipe Feed", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    #输出平均速度
    print(f"mean_v:{total_v / frame_idx:.2f}m/s")
    cap.release()
    cv2.destroyAllWindows()
