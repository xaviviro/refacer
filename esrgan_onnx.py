import numpy as np

class ESRGAN:

    def __init__(self, session=None):
        self.session = session
        self.model_input = self.session.get_inputs()[0].name

    def get(self, image_array):
        input_size = image_array.shape[1]
        image_array = image_array.transpose(2, 0, 1).astype('float32') / 255.0
        image_array = image_array.reshape(1, 3, image_array.shape[1], image_array.shape[2])
        result = self.session.run([], {self.model_input: image_array})[0][0]
        result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
        scale_factor = int(result.shape[1]/input_size)
        return result.astype(np.uint8), scale_factor