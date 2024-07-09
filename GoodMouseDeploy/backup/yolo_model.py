import aidlite


# TODO 需要设置模型编号？
class DetModel():
    def __init__(self):
        '''
        yolov8n
        input: float32 [1, 3, 256, 256] - 通道序: NCHW
        output: float32 [1, 5, 1344] - 5 : xywh + conf (1 class)
        '''
        #  input shape (1, 3, 256, 256) BCHW and output shape(s) (1, 5, 1344) 
        self.model_path = "models/yolov8/best_saved_model/best_float32.tflite" # "models/aidlux/palm_detection.tflite" "models/rtm/rtmdet.mnn" # "models/rtm/raw_tf_rtmdet_fp32.tflite" #"models/rtm/rtmdet_320_fp32.tflite" # "models/rtm/end2end.nb" # rtmdet_320_fp32.tflite"
        self.in_shape = [[1, 3, 256, 256]]
        self.out_shape = [[1, 5, 1344]] 

        # 构造 Model
        self.model = aidlite.Model.create_instance(self.model_path)
        if self.model is None:
            print("DetModel: Create model failed!")
            exit()
        self.model.set_model_properties(self.in_shape, aidlite.DataType.TYPE_FLOAT32, self.out_shape, aidlite.DataType.TYPE_FLOAT32)
        
        # 构造 Config
        self.config = aidlite.Config.create_instance()
        self.config.implement_type = aidlite.ImplementType.TYPE_FAST
        self.config.framework_type = aidlite.FrameworkType.TYPE_TFLITE
        self.config.accelerate_type = aidlite.AccelerateType.TYPE_CPU
        self.config.number_of_threads = 4

        # 构造 Interpreter
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
