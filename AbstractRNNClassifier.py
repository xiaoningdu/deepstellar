
class AbstractRNNClassifier:

    def load_hidden_state_model(self, model_path):
        pass

    def input_preprocess(self, data):
        return data

    def profile_train_data(self, profile_save_path):
        pass

    def get_state_profile(self, inputs):
        pass
