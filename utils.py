from models import *
import sys
from pytorch_model_summary import summary

existing_models = ["LeNet", "ResNet"]


def get_model(model_name):
    if str.lower(model_name) == 'lenet':
        model = LeNet()
        name = "LeNet"
    elif 'resnet' in str.lower(model_name):
        num_layers = int(model_name.lower().strip('resnet'))
        model = make_resnet(num_layers)
        name = "ResNet" + str(num_layers)
    else:
        print("We have not implemented model: " + model_name + " yet")
        print("Here's what we have:")
        print(existing_models)
        sys.exit()

    print(summary(model, torch.zeros(1, 3, 32, 32), show_input=True))
    print("Model ", name, "created...")
    return model


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)