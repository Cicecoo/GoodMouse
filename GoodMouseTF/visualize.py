import numpy as np
import cv2

def plot_detections(img, detections, with_keypoints=True):
        output_img = img
        print(img.shape)
        x_min=[0,0]
        x_max=[0,0]
        y_min=[0,0]
        y_max=[0,0]
        hand_nums=len(detections)
        print("Found %d hands" % hand_nums)
        if hand_nums >2:
            hand_nums=2        
        for i in range(hand_nums):
            ymin = detections[i][ 0] * img.shape[0]
            xmin = detections[i][ 1] * img.shape[1] 
            ymax = detections[i][ 2] * img.shape[0]
            xmax = detections[i][ 3] * img.shape[1]
            w=int(xmax-xmin)
            h=int(ymax-ymin)
            h=max(h,w)
            h=h*224./128.
            
            x=(xmin+xmax)/2.
            y=(ymin+ymax)/2.
            
            xmin=x-h/2.
            xmax=x+h/2.
            ymin=y-h/2.-0.18*h
            ymax=y+h/2.-0.18*h
            
            x_min[i]=int(xmin)
            y_min[i]=int(ymin)
            x_max[i]=int(xmax)
            y_max[i]=int(ymax)            
            p1 = (int(xmin),int(ymin))
            p2 = (int(xmax),int(ymax))
            cv2.rectangle(output_img, p1, p2, (0,255,255),2,1)
            
        return x_min,y_min,x_max,y_max
        
def draw_mesh(image, mesh, mark_size=4, line_width=1):
    """Draw the mesh on an image"""
    # The mesh are normalized which means we need to convert it back to fit
    # the image size.
    image_size = image.shape[0]
    mesh = mesh * image_size
    for point in mesh:
        cv2.circle(image, (point[0], point[1]),
                   mark_size, (255, 0, 0), 4)

# 计算手掌中心
def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手腕1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手腕2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 食指：根部
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：根部
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 无名指：根部
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：根部
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[0] * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image
    
def draw_landmarks(image, cx, cy, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    # 绘制关键点
    for index, landmark in enumerate(landmarks):
        # if landmark.visibility < 0 or landmark.presence < 0:
        #     continue

        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手腕1
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手腕2
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 拇指：根部
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 拇指：第1关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 拇指：指尖
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 食指：根部
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 食指：第2关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 食指：第1关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 食指：指尖
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：根部
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指尖
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 无名指：根部
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 无名指：第2关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 无名指：第1关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 无名指：指尖
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：根部
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1关节
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指尖
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 绘制连线
    if len(landmark_point) > 0:
        # 拇指
        cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)
        # 食指
        cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)
        # 中指
        cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv2.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv2.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)
        # 无名指
        cv2.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv2.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)
        # 小指
        cv2.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv2.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv2.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)
        # 手掌
        cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)

    return image