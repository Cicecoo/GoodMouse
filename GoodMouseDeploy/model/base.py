import aidlite

class AidliteBaseModel:
    def __init__(self, model_path, in_shape, out_shape, framework_type, accelerate_type, number_of_threads) -> None:
        self.model_path = model_path
        self.in_shape = in_shape
        self.out_shape = out_shape

        self.model = aidlite.Model.create_instance(self.model_path)
        if self.model is None:
            print(f"{self.__class__.__name__}: Create model failed!")
            exit()

        self.model.set_model_properties(self.in_shape, aidlite.DataType.TYPE_FLOAT32, self.out_shape, aidlite.DataType.TYPE_FLOAT32)

        self.config = aidlite.Config.create_instance()
        self.config.implement_type = aidlite.ImplementType.TYPE_FAST
        self.config.framework_type = framework_type
        self.config.accelerate_type = accelerate_type
        self.config.number_of_threads = number_of_threads

        self.interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(self.model, self.config)
        if self.interpreter is None:
            print(f"{self.__class__.__name__}: build_interpretper_from_model_and_config failed!")
            exit()
        retval = self.interpreter.init()
        if retval != 0:
            print(f"{self.__class__.__name__}: interpreter init failed!")
            exit()
        retval = self.interpreter.load_model()
        if retval != 0:
            print(f"{self.__class__.__name__}: interpreter load model failed!")
            exit()
        print(f"{self.__class__.__name__}: model load success!")
    
    def preprocess(self, image):
        pass

    def inference(self, image):
        pass

    def postprocess(self, outputs):
        pass

    def visualize(self, image, outputs):
        pass


 