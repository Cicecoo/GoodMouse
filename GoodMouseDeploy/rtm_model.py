import aidlite


# TODO 需要设置模型编号？
class DetModel():
    def __init__(self):
        '''
        rtmdet 
        来自 RTMPose 开源，与此处（https://mmyolo.readthedocs.io/zh-cn/latest/recommended_topics/algorithm_descriptions/rtmdet_description.html）描述似乎不同
        input: float32 1, 3, 320, 320] - channel order: ?
        output: float32 [1, Gatherdets_dim_1, 5], int64 [1, Gatherdets_dim_1] - dets, labels
        '''
        self.model_path = "models/rtmdet/rtmdet_.mnn" # "models/aidlux/palm_detection.tflite" "models/rtm/rtmdet.mnn" # "models/rtm/raw_tf_rtmdet_fp32.tflite" #"models/rtm/rtmdet_320_fp32.tflite" # "models/rtm/end2end.nb" # rtmdet_320_fp32.tflite"
        self.in_shape = [[1, 3, 320, 320]] # [[1, 128, 128 ,3]] 
        self.out_shape = [[1, 200, 5], [1, 200]] # [[1, 896, 18], [1, 896, 1]]

        self.model = aidlite.Model.create_instance(self.model_path)
        if self.model is None:
            print("DetModel: Create model failed!")
            exit()

        self.model.set_model_properties(self.in_shape, aidlite.DataType.TYPE_FLOAT32, self.out_shape, aidlite.DataType.TYPE_FLOAT32)

        self.config = aidlite.Config.create_instance()
        self.config.implement_type = aidlite.ImplementType.TYPE_FAST
        self.config.framework_type = aidlite.FrameworkType.TYPE_MNN
        self.config.accelerate_type = aidlite.AccelerateType.TYPE_CPU
        self.config.number_of_threads = 4

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(self.model, self.config)
        if self.interpreter is None:
            print("DetModel: build_interpretper_from_model_and_config failed!")
            exit()
        retval = self.interpreter.init()
        if retval != 0:
            print(f"DetModel: interpreter init failed!")
            exit()
        retval = self.interpreter.load_model()
        if retval != 0:
            print("DetModel: interpreter load model failed!")
            exit()
        print("DetModel: detect model load success!")


class PoseModel():
    def __init__(self):
        '''
        rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256
        input: float32 [batch, 3, 256, 256]
        output: float32 [batch, MatMulsimcc_x_dim_1, 512], float32 [batch, MatMulsimcc_x_dim_1, 512] - simcc_x, simcc_y
        '''
        self.model_path = "weights/pose/rtmpose.mnn"
        self.model_input_size = (256, 256)
        
        # 输入输出结构通过 https://netron.app/ 可视化得到，参考例程修改
        self.in_shape = [[1, 3, 256, 256]] # 注意 w h c 的顺序，此处channel优先，与例程不同
        self.out_shape = [[1, 21, 512], [1, 21, 512]] # simcc_x, simcc_y

        # 创建Model实例对象，并设置模型相关参数
        self.model = aidlite.Model.create_instance(self.model_path)
        if self.model is None:
            print("PoseModel: create model failed!")

        # 设置模型属性
        self.model.set_model_properties(self.in_shape, aidlite.DataType.TYPE_FLOAT32, self.out_shape, aidlite.DataType.TYPE_FLOAT32)
        
        # 创建Config实例对象，并设置配置信息
        self.config = aidlite.Config.create_instance()
        self.config.implement_type = aidlite.ImplementType.TYPE_FAST
        self.config.framework_type = aidlite.FrameworkType.TYPE_MNN
        self.config.accelerate_type = aidlite.AccelerateType.TYPE_GPU
        self.config.number_of_threads = 4

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(self.model, self.config)
        if self.interpreter is None:
            print("PoseModel: build_interpretper_from_model_and_config failed!")
            exit()
        result = self.interpreter.init()
        if result != 0:
            print(f"PoseModel: interpreter init failed!")
            exit()
        result = self.interpreter.load_model()
        if result != 0:
            print("PoseModel: interpreter load model failed!")
            exit()
        print("PoseModel: pose detect model load success!")