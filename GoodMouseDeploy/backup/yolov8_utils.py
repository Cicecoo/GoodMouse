import cv2
import numpy as np

# 置信度
confidence_thres = 0.35
# iou阈值
iou_thres = 0.1
# 类别
classes = {0: 'hand',}
# 随机颜色
color_palette = np.random.uniform(100, 255, size=(len(classes), 3))

def calculate_iou(box, other_boxes):
    """
    计算给定边界框与一组其他边界框之间的交并比（IoU）。

    参数：
    - box: 单个边界框，格式为 [x1, y1, width, height]。
    - other_boxes: 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。

    返回值：
    - iou: 一个数组，包含给定边界框与每个其他边界框的IoU值。
    """

    # 计算交集的左上角坐标
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    # 计算交集的右下角坐标
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    # 计算交集区域的面积
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # 计算给定边界框的面积
    box_area = box[2] * box[3]
    # 计算其他边界框的面积
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    # 计算IoU值
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou

def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(filtered_boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(filtered_scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices


def draw_detections(img, box, score, class_id):
    """
    在输入图像上绘制检测到的对象的边界框和标签。

    参数:
            img: 要在其上绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。

    返回:
            无
    """

    # 提取边界框的坐标
    x1, y1, w, h = box
    # 根据类别ID检索颜色
    color = color_palette[class_id]
    # 在图像上绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    # 创建标签文本，包括类名和得分
    label = f'{classes[class_id]}: {score:.2f}'
    # 计算标签文本的尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算标签文本的位置
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    # 绘制填充的矩形作为标签文本的背景
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def preprocess_image_for_tflite(image):
        image = cv2.resize(image, (320, 320))
        image = image / 255.0
        # image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        # print(image.shape)
        return image

def yolo_preprocess(img, input_width, input_height):
    """
    在执行推理之前预处理输入图像。

    返回:
        image_data: 为推理准备好的预处理后的图像数据。
    """

    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]
    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将图像大小调整为匹配输入形状
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # image_data = np.expand_dims(image_data, axis=0).astype('float32')
    # 返回预处理后的图像数据
    return image_data, img_height, img_width

def apply_nms(data, score_threshold=0.1, nms_threshold=0.5, class_id=0): # 目前是单类别进行，如果多类别要复用，效率低
        # 提取中心xy和wh
        center_x = data[:, 0]
        center_y = data[:, 1]
        w = data[:, 2]
        h = data[:, 3]
        conf = data[:, class_id+4] # 提取置信度，类别为0时提取0+4列
        # 计算边界框的四个角的位置
        x_min = center_x - (w / 2)
        y_min = center_y - (h / 2)
        x_max = center_x + (w / 2)
        y_max = center_y + (h / 2)
        # 把它们放入一个列表中
        boxes = np.array([x_min, y_min, x_max, y_max]).T.tolist()
        # 使用opencv的nms
        indices = cv2.dnn.NMSBoxes(boxes, conf, score_threshold, nms_threshold)
        final_boxes = [boxes[i] for i in indices]
        # print(final_boxes)
        # 把conf加入到最后一列
        for i in range(len(final_boxes)):
            final_boxes[i].append(conf[indices[i]])
        return final_boxes

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2 ):
        width = image.shape[1]
        height = image.shape[0]
        for box in boxes:
            pt1 = (int(box[0]*width), int(box[1]*height)) # 左上角，记得反归一化并转换为int
            pt2 = (int(box[2]*width), int(box[3]*height)) # 右下角
            cv2.rectangle(image, pt1, pt2, color, thickness)
            # 画置信度,保留两位小数
            cv2.putText(image, str(round(box[4], 2)), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

class_num = 1
output_length = 1344
def yolo_postprocess(input_image, output, input_width, input_height, img_width, img_height):
    """
    对模型输出进行后处理，提取边界框、得分和类别ID。

    参数:
        input_image (numpy.ndarray): 输入图像。
        output (numpy.ndarray): 模型的输出。
        input_width (int): 模型输入宽度。
        input_height (int): 模型输入高度。
        img_width (int): 原始图像宽度。
        img_height (int): 原始图像高度。

    返回:
        numpy.ndarray: 绘制了检测结果的输入图像。
    """

    # 转置和压缩输出以匹配预期的形状
    # outputs = np.squeeze(output[0])
    # outputs = np.transpose()
    
    # outputs = output.reshape((class_num+4, output_length)).T

    boxes = apply_nms(output)
    # print(boxes, '\n')
    draw_boxes(input_image, boxes)

    # 获取输出数组的行数
    # rows = outputs.shape[0]
    # # 用于存储检测的边界框、得分和类别ID的列表
    # boxes = []
    # scores = []
    # class_ids = []
    # # 计算边界框坐标的缩放因子
    # x_factor = img_width / input_width
    # y_factor = img_height / input_height
    # # 遍历输出数组的每一行
    # for i in range(rows):
    #     # 从当前行提取类别得分
    #     classes_scores = outputs[i][4:]
    #     # 找到类别得分中的最大得分
    #     max_score = np.amax(classes_scores)

    #     # print('max score:', max_score)
    #     # print(outputs[i])

    #     # if(i > 10):
    #     #     exit()


    #     # 如果最大得分高于置信度阈值
    #     if max_score >= confidence_thres:

    #         print('max score:', max_score)
    #         print(outputs[i])

    #         # 获取得分最高的类别ID
    #         class_id = np.argmax(classes_scores)
    #         # 从当前行提取边界框坐标
    #         x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
    #         # 计算边界框的缩放坐标
    #         left = int((x - w / 2) * x_factor)
    #         top = int((y - h / 2) * y_factor)
    #         width = int(w * x_factor)
    #         height = int(h * y_factor)
    #         # 将类别ID、得分和框坐标添加到各自的列表中
    #         class_ids.append(class_id)
    #         scores.append(max_score)
    #         boxes.append([left, top, width, height])
    # # 应用非最大抑制过滤重叠的边界框
    # indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # # 遍历非最大抑制后的选定索引
    # for i in indices:
    #     # 根据索引获取框、得分和类别ID
    #     box = boxes[i]
    #     score = scores[i]
    #     class_id = class_ids[i]
    #     # 在输入图像上绘制检测结果
    #     draw_detections(input_image, box, score, class_id)
    # # 返回修改后的输入图像
    # return input_image
