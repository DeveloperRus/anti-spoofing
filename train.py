import torch
import argparse
import os
import torchvision

from tqdm import tqdm
from SZCVI_model import SZCVI
from common_functions import *
from sklearn.model_selection import train_test_split
from dataset import AntispoofDataset


def fit_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train(True)

    running_loss = 0.0
    running_corrects = 0
    processed_data = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = []
        for i in range(5):
            predictions.append([model(batch[:, i])])

        predictions = sum(model(batch[:, i].to(device)) for i in range(5)) / 5
        loss = loss_fn(predictions, labels)

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(predictions, 1)
        running_loss += loss.item() * batch.size(0)
        running_corrects += torch.sum(predictions == labels.data)
        processed_data += batch.size(0)

    train_loss = running_loss / processed_data
    train_acc = running_corrects.cpu().numpy() / processed_data

    return train_loss, train_acc


def eval_epoch(model, data_loader, loss_fn, device):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for batch, labels in data_loader:
        batch = batch.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            predictions = sum(model(batch[:, i].to(device)) for i in range(5)) / 5
            loss = loss_fn(predictions, labels)
            predictions = torch.argmax(predictions, 1)

        running_loss += loss.item() * batch.size(0)
        running_corrects += torch.sum(predictions == labels.data)
        processed_size += batch.size(0)

    loss = running_loss / processed_size
    acc = running_corrects.double() / processed_size

    return loss, acc


def train(data_loader, model, epochs, device, save_path=None):
    history = []
    log_template = "\nLog epoch {ep:03d}:\n\ttrain_loss: {t_loss:0.4f} \n\tval_loss {v_loss:0.4f} \n\ttrain_acc " \
                   "{t_acc:0.4f} \n\tval_acc {v_acc:0.4f}"

    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train_loss, train_acc = fit_epoch(model, data_loader['train'], loss_fn, optimizer, device)
            val_loss, val_acc = eval_epoch(model, data_loader['val'], loss_fn, device)

            history.append((train_loss, train_acc, val_loss, val_acc))
            tqdm.write(log_template.format(ep=epoch + 1, t_loss=train_loss,
                                           v_loss=val_loss, t_acc=train_acc,
                                           v_acc=val_acc))
            pbar_outer.update(1)

            if save_path is not None:
                save_weights(model, save_path, val_acc.item())

    return history


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--module-name', type=str, required=True)
    parser.add_argument('--path-weights', type=str, required=False)
    parser.add_argument('--path-train-dir', type=str, required=True)
    args = parser.parse_args()

    # Upload video paths
    path_videos = []

    for label in ['2dmask', 'real', 'printed', 'replay']:
        videos = os.listdir(os.path.join(args.path_train_dir, label))

        for video in videos:
            if not os.path.isdir(os.path.join(args.path_train_dir, label, video)):
                continue

            path_videos.append({
                'path': os.path.join(args.path_train_dir, label, video),
                'label': int(label != 'real'),
            })

    X_train, X_test = train_test_split(path_videos, test_size=0.2, random_state=123)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = {
        'train': AntispoofDataset(
            X_train, transform=transform),
        'val': AntispoofDataset(
            X_test, transform=transform)}

    data_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=10, shuffle=True, num_workers=4)
        for x in ['train', 'val']}

    if args.module_name == 'SZCVI':
        model = SZCVI()
    else:
        raise NameError

    if args.path_weights is not None:
        load_weights(model, args.path_weights, DEVICE)

    history = train(data_loaders, model=model, epochs=1, device=DEVICE, save_path='weights')
