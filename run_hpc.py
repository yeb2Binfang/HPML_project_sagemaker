import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import argparse
import os
import random
import numpy as np

from utilization import getUtilization

from time import perf_counter


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model, device, test_loader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def main(num_epochs_default, model):
    # num_epochs_default = 20
    batch_size_default = 128  # 1024
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "checkpoint"
    model_filename_default = "densenet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=num_epochs_default)
    parser.add_argument("--batch_size", type=int, help="Training batch size for one process.",
                        default=batch_size_default)
    parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=learning_rate_default)
    parser.add_argument("--random_seed", type=int, help="Random seed.", default=random_seed_default)
    parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default=model_dir_default)
    parser.add_argument("--model_filename", type=str, help="Model filename.", default=model_filename_default)
    parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
    argv = parser.parse_args()

    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    '''
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    '''

    # We need to use seeds to make sure that the models initialized in different processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    # torch.distributed.init_process_group(backend="gloo")

    # Encapsulate the model on the GPU assigned to the current process
    # model = torchvision.models.resnet18(pretrained=False)
    # model = torchvision.models.vgg16(pretrained=False)
    # model = torchvision.models.mobilenet_v2(pretrained=False)
    # model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
    # model = torchvision.models.densenet161(pretrained=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = net = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    # Test loader does not have to follow distributed sampling strategy
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    best_acc = 0.0

    # Loop over the dataset multiple times
    for epoch in range(num_epochs):

        # Save and evaluate model routinely
        accuracy = evaluate(model=model, device=device, test_loader=test_loader)

        print("-" * 75)
        print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
        print("-" * 75)

        best_acc = max(best_acc, accuracy)

        model.train()

        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return best_acc


if __name__ == "__main__":
    net_names = ['ResNet18', 'VGG16', 'MobileNet', 'ShuffleNet', 'DenseNet']
    nets = [torchvision.models.resnet18(pretrained=False), torchvision.models.vgg16(pretrained=False),
            torchvision.models.mobilenet_v2(pretrained=False), torchvision.models.shufflenet_v2_x1_0(pretrained=False),
            torchvision.models.densenet161(pretrained=False)]
    for i in range(len(net_names)):
        num_epochs = 20
        start_training_time = perf_counter()
        accuracy = main(num_epochs, nets[i])
        end_training_time = perf_counter()
        print("Total time: %.6f s" % ((end_training_time - start_training_time) / num_epochs))
