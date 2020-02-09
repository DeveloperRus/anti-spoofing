import argparse
import os, ssl

import torchvision
from torchvision.models import vgg16_bn
from common_functions import *
from dataset import AntispoofDataset


class SZCVI(torch.nn.Module):
    """
    The video frames were resized into 224×224 images and fed into a CNN model. The architecture of this model consists
    of five convolutional layers and one fully connected layer. The convolutional layers were inspired by the VGG model.
    The scores of the sampled frames were averaged to obtain the final score for each video file.
    """
    def __init__(self):
        super().__init__()

        self.module = torch.nn.Sequential(
            *(list(vgg16_bn(pretrained=True).features[:17])),
            Flatten(),
            torch.nn.Linear(256 * 56 * 56, 2)
        )

    def forward(self, x):
        return self.module(x)


if __name__ == '__main__':
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Read args
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-weights', type=str, required=True)
    parser.add_argument('--path-test-dir', type=str, required=True)
    args = parser.parse_args()

    # Prepare model
    model = SZCVI().to(DEVICE)

    load_weights(model, args.path_weights, DEVICE)

    # Upload video paths
    path_videos = []

    videos = os.listdir(args.path_test_dir)

    for video in videos:
        if not os.path.isdir(os.path.join(args.path_test_dir, video)):
            continue

        path_videos.append({
            'path': os.path.join(args.path_test_dir, video)
        })

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = AntispoofDataset(path_videos, mode='test', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    # Predict
    predictions = predict(model, dataloader, DEVICE)
    print(predictions)
