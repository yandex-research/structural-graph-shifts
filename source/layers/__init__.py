from .utils import model_name_to_class

def get_model_class(model_class_name):
    return model_name_to_class[model_class_name]