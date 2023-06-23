import numpy as np

class ESRGAN:
    def __init__(self, session):
        self.session = session
        self.model_input = self.session.get_inputs()[0].name

    def _pre_process(self, image_array):
        image_array = image_array.transpose(2, 0, 1).astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def _post_process(self, result):
        result = np.clip(result.transpose(1, 2, 0), 0, 1) * 255.0
        return result.astype(np.uint8)

    def get(self, image_array):
        input_size = image_array.shape[1]
        image_array = self._pre_process(image_array)
        ort_inputs = {self.model_input: image_array}
        result = self.session.run(None, ort_inputs)[0][0]
        result = self._post_process(result)
        scale_factor = int(result.shape[1] / input_size)
        return result, scale_factor
