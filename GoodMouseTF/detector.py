import numpy as np
import tflite_runtime.interpreter as tflite


class HandDetector():
    def __init__(self):
        self.model_path = 'models/palm_detection.tflite'
        self.in_shape = [1, 128, 128, 3]
        self.out_shape = [[1, 896, 18], [1, 896, 1]]

        self.interpreter = tflite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.width = self.input_details[0]['shape'][1]
        self.height = self.input_details[0]['shape'][2]

    def inference(self, img):
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype('float32')
        img = (2.0 / 255.0) * img - 1.0
        img = np.expand_dims(img, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        raw_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        classificators = self.interpreter.get_tensor(self.output_details[1]['index'])
        return raw_boxes, classificators


# model = aidlite.Model.create_instance(model_path)
# if model is None:
#     print("Create model failed !")

# # 设置模型属性
# model.set_model_properties(inShape, aidlite.DataType.TYPE_FLOAT32, outShape,aidlite.DataType.TYPE_FLOAT32)
# # 创建Config实例对象，并设置配置信息
# config = aidlite.Config.create_instance()
# config.implement_type = aidlite.ImplementType.TYPE_FAST
# config.framework_type = aidlite.FrameworkType.TYPE_TFLITE
# config.accelerate_type = aidlite.AccelerateType.TYPE_GPU
# config.number_of_threads = 8
# # 创建推理解释器对象
# fast_interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
# if fast_interpreter is None:
#     print("build_interpretper_from_model_and_config failed !")
# # 完成解释器初始化
# result = fast_interpreter.init()
# if result != 0:
#     print(f"interpreter init failed !")
# # 加载模型
# result = fast_interpreter.load_model()
# if result != 0:
#     print("interpreter load model failed !")
# print("detect model load success!")



model_path1="models/hand_landmark.tflite"
inShape1 =[[1 , 224 , 224 ,3]]
outShape1= [[1 , 63],[1],[1]]
# 创建Model实例对象，并设置模型相关参数
model1 = aidlite.Model.create_instance(model_path1)
if model1 is None:
    print("Create model failed !")
# 设置模型属性
model1.set_model_properties(inShape1, aidlite.DataType.TYPE_FLOAT32, outShape1,
                           aidlite.DataType.TYPE_FLOAT32)
# 创建Config实例对象，并设置配置信息
config1 = aidlite.Config.create_instance()
config1.implement_type = aidlite.ImplementType.TYPE_FAST
config1.framework_type = aidlite.FrameworkType.TYPE_TFLITE
config1.accelerate_type = aidlite.AccelerateType.TYPE_CPU
config.number_of_threads = 4
# 创建推理解释器对象
fast_interpreter1 = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model1, config1)
if fast_interpreter1 is None:
    print("build_interpretper_from_model_and_config failed !")
# 完成解释器初始化
result = fast_interpreter1.init()
if result != 0:
    print(f"interpreter init failed !")
# 加载模型
result = fast_interpreter1.load_model()
if result != 0:
    print("interpreter load model failed !")
print("detect model load success!")


anchors = np.load('models/anchors.npy').astype(np.float32)

aidlux_type="root"
















bi_hand=False

x_min=[0,0]
x_max=[0,0]
y_min=[0,0]
y_max=[0,0]

fface=0.0
use_brect=True