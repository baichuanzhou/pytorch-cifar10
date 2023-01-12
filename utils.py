import models
import sys
from pytorch_model_summary import summary
import torch
import os
import torch.nn.functional as F

existing_models = ["LeNet", "ResNet", "GoogLeNet"]


def get_model(model_name, print_info=True):
    if str.lower(model_name) == 'lenet':
        model = models.LeNet()
        name = "LeNet"
    elif 'resnet' in str.lower(model_name):
        num_layers = int(model_name.lower().strip('resnet'))
        model = models.make_resnet(num_layers)
        name = "ResNet" + str(num_layers)
    elif 'googlenet' in str.lower(model_name):
        model = models.GoogLeNet()
        name = "GoogLeNet"
    else:
        print("We have not implemented model: " + model_name + " yet")
        print("Here's what we have:")
        print(existing_models)
        sys.exit()
    if print_info:
        print(summary(model, torch.zeros(1, 3, 32, 32), show_input=True))
        print("Model ", name, "created...")
    return model, name


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


def train(model,
          optimizer,
          loader_train,
          loader_val,
          loss_his,
          acc_his,
          epoch=10,
          lr_scheduler=None,
          device='cuda'):
    model.to(device=device)
    optimizer_to(optimizer, device)
    for e in range(epoch):
        for i, (x, y) in enumerate(loader_train):
            model.train()  # Set the model to training mode

            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)

            criterion = F.cross_entropy

            loss = criterion(scores, y)

            # Zero out the gradients before so that it can take the next step
            optimizer.zero_grad()

            # Backward pass so that losses and gradients can flow through the computational graph
            loss.backward()
            # Update the gradients and takes a step
            optimizer.step()

            if e == 0 and i == 0:
                print("Initial loss: %.2f" % loss.item())  # Check if the initial loss is log(num_classes)
            if i % 80 == 0:
                print("Epoch %d with iteration %d, current loss: %.2f" % (e, i, loss.item()))

            loss_his.append(loss.item())

        print(f"Training with Epoch {e}")
        if lr_scheduler is not None:
            lr_scheduler.step()
        acc = check_accuracy(model, loader_val)
        acc_his.append(acc)


def check_accuracy(model, loader, device='cuda'):
    model.to(device=device)
    if loader.dataset.train:
        print("Checking on validation set...")
    else:
        print("Checking on test set...")
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += y.size(0)
        acc = float(num_correct) / num_samples
        print("Got %d / %d correct rates: %.2f" % (num_correct, num_samples, 100 * acc) + "%")
    return acc


def save(name,
         model,
         optimizer,
         acc_his,
         loss_his,
         epochs):
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if os.path.isfile('./checkpoints/%s.pth' % name):
        checkpoint = torch.load('./checkpoints/%s.pth' % name)
        checkpoint['params'] = model.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['accuracy_history'] = acc_his
        checkpoint['loss_history'] = loss_his
        checkpoint['epoch'] += epochs
    else:
        checkpoint = {
            'params': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'accuracy_history': acc_his,
            'loss_history': loss_his,
            'epoch': epochs
        }
    print("Saving %s's parameters" % name)
    torch.save(checkpoint, './checkpoints/%s.pth' % name)


def load(model_name, device='cuda'):
    model, model_name = get_model(model_name)
    if not os.path.isdir('checkpoints'):
        raise FileNotFoundError('No checkpoints directory')
    if os.path.isfile('./checkpoints/%s.pth' % model_name):
        checkpoint = torch.load('./checkpoints/%s.pth' % model_name)
        model.load_state_dict(checkpoint['params'])
    else:
        raise FileNotFoundError('No such models')
    model.to(device=device)
    return model
