# from .model import CosCNN, MyCustomModel
from .model import CosCNN
from .CustomModel import CustomModel

MODEL_REGISTRY = {
    'CosCNN': CosCNN,
    'MyCustomModel': CustomModel,
}