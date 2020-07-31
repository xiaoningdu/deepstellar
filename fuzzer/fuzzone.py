from keras import backend as K
import numpy as np
def predict(self, input_data):
    inp = self.model.input
    functor = K.function([inp] + [K.learning_phase()], self.outputs)
    outputs = functor([input_data, 0])
    return outputs
def fetch_function(handler, input_batches, preprocess):
    _, img_batches, _ = input_batches
    if len(img_batches) == 0:
        return None, None
    preprocessed = preprocess(img_batches)
    outputs = handler.predict(preprocessed)
    return outputs[1], np.expand_dims(np.argmax(outputs[0], axis=1),axis=0)


def build_fetch_function(handler, preprocess):
    def func(input_batches):

        return fetch_function(
            handler,
            input_batches,
            preprocess
        )
    return func


def adptive_coverage_function(handler, cov_num):
    def func(layerouts):
        """The fetch function."""
        ptr = np.zeros(cov_num, dtype=np.uint8)
        return handler.update_coverage(layerouts, ptr)

    return func
