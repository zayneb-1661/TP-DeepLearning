import os
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

# ============================================================
# 1. Reproductibilité
# ============================================================

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
random.seed(0)


# ============================================================
# 2. Hyperparamètres
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--batch-size", type=int, default=32)
args = parser.parse_args()

hparams = dict(
    model="MLP",
    batch_size=args.batch_size,
    lr=args.lr,
    seed=0,
    weight_decay=0.0
)

run_name = (
    f"{hparams['model']}"
    f"_lr{hparams['lr']}"
    f"_bs{hparams['batch_size']}"
    f"_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

logdir = os.path.join("runs", run_name)

print("logdir:", logdir)

writer = SummaryWriter(log_dir=logdir)


# ============================================================
# 3. Dataset CIFAR-10
# ============================================================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

trainset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

testset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 4. Split train / validation
# ============================================================

N = len(trainset)

val_size = int(0.1 * N)
train_size = N - val_size

generator = torch.Generator().manual_seed(hparams["seed"])

train_subset, val_subset = random_split(
    trainset,
    [train_size, val_size],
    generator=generator
)


def get_num_workers(default=1, cap=1):
    try:
        n = int(os.getenv("SLURM_CPUS_PER_TASK", default))
    except Exception:
        n = default
    return max(0, min(cap, n))


num_workers = get_num_workers()

trainloader = DataLoader(
    train_subset,
    batch_size=hparams["batch_size"],
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

valloader = DataLoader(
    val_subset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

testloader = DataLoader(
    testset,
    batch_size=hparams["batch_size"],
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


# ============================================================
# 5. Modèle
# ============================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================
# 6. Device, modèle, loss et optimizer
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=hparams["lr"],
    momentum=0.9,
    weight_decay=hparams["weight_decay"]
)


# ============================================================
# 7. Fonction de calcul des métriques
# ============================================================

@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        loss_sum += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    loss_avg = loss_sum / total
    acc = correct / total

    return loss_avg, acc


# ============================================================
# 8. Entraînement + TensorBoard
# ============================================================

global_step = 0
EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()

    running_loss_sum = 0.0
    running_total = 0

    for b, (x, y) in enumerate(trainloader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()

        # Logging tous les 10 batches
        if b % 10 == 0:
            writer.add_scalar(
                "Loss/train_step",
                loss.item(),
                global_step
            )

        optimizer.step()

        running_loss_sum += loss.item() * y.size(0)
        running_total += y.size(0)

        global_step += 1

    # Métriques de fin d'époque
    train_loss = running_loss_sum / running_total

    val_loss, val_acc = epoch_metrics(
        valloader,
        model,
        criterion,
        device
    )

    # Logging TensorBoard
    writer.add_scalar(
        "Loss/train",
        train_loss,
        epoch
    )

    writer.add_scalar(
        "Loss/val",
        val_loss,
        epoch
    )

    writer.add_scalar(
        "Accuracy/val",
        val_acc,
        epoch
    )

    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.3f}"
    )


# ============================================================
# 9. Écriture des logs
# ============================================================

writer.flush()
writer.close()

print("Training finished")
print("TensorBoard logs saved in:", logdir)


# ============================================================
# 10. Évaluation finale sur test
# ============================================================

test_loss, test_acc = epoch_metrics(
    testloader,
    model,
    criterion,
    device
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")


# ============================================================
# 11. Sauvegarde
# ============================================================

os.makedirs("outputs", exist_ok=True)

torch.save(
    model.state_dict(),
    f"outputs/{run_name}.pth"
)

print(f"Model saved: outputs/{run_name}.pth")

