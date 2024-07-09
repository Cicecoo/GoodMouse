import numpy as np
import time
from typing import List, Tuple

import aidlite
from model.base import AidliteBaseModel

from rtmpose_utils import bbox_xyxy2cs, top_down_affine, get_simcc_maximum, decode, visualize


class RTMPoseModel(AidliteBaseModel):
    def __init__(self):
        model_path = "weights/pose/rtmpose.mnn"
        in_shape = [[1, 3, 256, 256]]
        out_shape = [[1, 21, 512], [1, 21, 512]]
        framework_type = aidlite.FrameworkType.TYPE_MNN
        accelerate_type = aidlite.AccelerateType.TYPE_GPU
        number_of_threads = 4
        super().__init__(model_path, in_shape, out_shape, framework_type, accelerate_type, number_of_threads)

    def __call__(self, image):
        return self.inference(image)
    
    def __str__(self):
        return "RTMPoseModel"

    def preprocess(self,
        img: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            input_size (tuple): Input image size in shape (w, h).

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        # get shape of image
        img_shape = img.shape[:2]
        bbox = np.array([0, 0, img_shape[1], img_shape[0]])

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)

        # do affine transformation
        resized_img, scale = top_down_affine(input_size, scale, center, img)


        # cv2.imshow("resized_img", resized_img)
        # print("resized_img:", resized_img.shape)

        # normalize image
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std

        # cicecoo
        # 注意要显式指定数据类型，遇到输出为 nan 时可以考虑是不是此处有问题
        resized_img = resized_img.astype('float32')

        return resized_img, center, scale
    
    def postprocess(self, outputs: List[np.ndarray],
                    model_input_size: Tuple[int, int],
                    center: Tuple[int, int],
                    scale: Tuple[int, int],
                    simcc_split_ratio: float = 2.0
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # use simcc to decode
        simcc_x, simcc_y = outputs
        keypoints, scores = decode(simcc_x, simcc_y, simcc_split_ratio)

        # rescale keypoints
        keypoints = keypoints / model_input_size * scale + center - scale / 2

        return keypoints, scores
    
    def inference(self, image):
        resized_img, center, scale = self.preprocess(image, (256, 256))

        self.interpreter.set_input_tensor(in_tensor_idx=0 , input_data=resized_img.data)
        self.interpreter.invoke()
        simcc_x = self.interpreter.get_output_tensor(0)
        simcc_y = self.interpreter.get_output_tensor(1)
        # print('simcc_x:', simcc_x.shape)
        # (10752,) to (1,21,512)
        simcc_x = simcc_x.reshape(1, 21, 512)
        simcc_y = simcc_y.reshape(1, 21, 512)
        outputs = (simcc_x, simcc_y)
        # first 21 x
        # print('simcc_x:', simcc_x.shape)

        keypoints, scores = self.postprocess(outputs, self.model_input_size, center, scale)
        return keypoints, scores
    
    def visualize(self, image, outputs):
        keypoints, scores = outputs
        return visualize(image, keypoints, scores, thr=0.3)
        
        