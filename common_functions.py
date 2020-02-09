import torch
import copy

from time import gmtime, strftime


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()


def save_weights(model, path, val_acc):
    name = strftime("%Yy.%mm.%dd.%Hh.%Mm", gmtime())
    model_weights = copy.deepcopy(model.state_dict())
    torch.save(model_weights, f"{path}/{round(val_acc, 3)}___{name}.pth")


def predict(model, data_loader, device):
    model.eval()
    res = []

    with torch.set_grad_enabled(False):
        for batch in data_loader:
            predictions = torch.argmax(sum(model(batch[:, i].to(device)) for i in range(5)) / 5, 1)
            res.append(predictions)

    return res