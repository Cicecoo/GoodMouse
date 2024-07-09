import aidlite
from model.base import AidliteBaseModel

class RTMDetModel(AidliteBaseModel):
    def __init__(self):
        model_path = "models/rtmdet/rtmdet_.mnn"
        in_shape = [[1, 3, 320, 320]]
        out_shape = [[1, 200, 5], [1, 200]]
        framework_type = aidlite.FrameworkType.TYPE_MNN
        accelerate_type = aidlite.AccelerateType.TYPE_CPU
        number_of_threads = 4
        super().__init__(model_path, in_shape, out_shape, framework_type, accelerate_type, number_of_threads)