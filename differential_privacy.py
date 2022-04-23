import warnings
warnings.simplefilter("ignore")

MAX_GRAD_NORM = [0.1, 0.5, 1, 5, 10]
EPSILON = [1, 5, 10, 50]
DELTA = [1e-5, 1e-4, 1e-3]
EPOCHS = 25

LR = 1e-3

BATCH_SIZE = 2048
MAX_PHYSICAL_BATCH_SIZE = 1024

import torch
import torchvision
import torchvision.transforms as transforms


#Can be calculated using the stats.py file
spec_MEAN = (0.3163, 0.1469, 0.4218)
spec_STD_DEV = (0.1903, 0.2710, 0.1723)


#mean and std: 
     #tensor([0.3163, 0.1469, 0.4218]) tensor([0.1903, 0.2710, 0.1723])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(spec_MEAN, spec_STD_DEV),
])

DATA_ROOT = '/dgxhome/sxb701/Hackallenge/Phase2_Ours/images'
TEST_DATA = '/dgxhome/sxb701/Hackallenge/Phase2_Ours/test'
train_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = datasets.ImageFolder(TEST_DATA, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

from torchvision import models

model = models.resnet18(num_classes=2)

from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)
errors[-5:]

model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)


device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model = model.to(device)

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)

def accuracy(preds, labels):
    return (preds == labels).mean()

from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager

def train(model, train_loader, optimizer, epoch, device):
        model.train()
        criterion = nn.CrossEntropyLoss()

        losses = []
        top1_acc = []
        
        with BatchMemoryManager(
            data_loader=train_loader, 
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
            optimizer=optimizer
        ) as memory_safe_data_loader:

            for i, (images, target) in enumerate(memory_safe_data_loader):   
                optimizer.zero_grad()
                images = images.to(device)
                target = target.to(device)

                # compute output
                output = model(images)
                loss = criterion(output, target)

                preds = np.argmax(output.detach().cpu().numpy(), axis=1)
                labels = target.detach().cpu().numpy()

                # measure accuracy and record loss
                acc = accuracy(preds, labels)

                losses.append(loss.item())
                top1_acc.append(acc)

                loss.backward()
                optimizer.step()

                if (i+1) % 200 == 0:
                    epsilon = privacy_engine.get_epsilon(DELTA)
                    print(
                        f"\tTrain Epoch: {epoch} \t"
                        f"Loss: {np.mean(losses):.6f} "
                        f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                        f"(ε = {epsilon:.2f}, δ = {DELTA})"
                    )

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

for m in range(len(MAX_GRAD_NORM)):
    for e in range(len(EPSILON)):
        for d in range(len(DELTA)):
    
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=EPOCHS,
                target_epsilon=EPSILON[e],
                target_delta=DELTA[d],
                max_grad_norm=MAX_GRAD_NORM[m])

            print(f"Using sigma={optimizer.noise_multiplier}, C={MAX_GRAD_NORM[m]}, D={DELTA[d]}, E = {EPSILON[e]}")

            from tqdm import tqdm

            for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
                train(model, train_loader, optimizer, epoch + 1, device)

            top1_acc = test(model, test_loader, device)
            #torch.cuda.empty_cache()
            #learn.destroy()
