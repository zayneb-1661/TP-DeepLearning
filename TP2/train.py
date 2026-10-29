import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# (Importez votre CardioDataset et vos loaders ici)
from torch.utils.data import DataLoader, random_split

from dataset import CardioDataset
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Initialisation Dataset et DataLoaders

dataset = CardioDataset("data/cardio_train.csv")

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# Modèle MLP

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(input_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),

            nn.Linear(hidden_size, 1),

            nn.Sigmoid()  # Sortie binaire [0, 1]
        )

    def forward(self, x):

        return self.net(x)


# Initialisation
# IMPORTANT : avant train_model()

batch = next(iter(train_loader))

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device :", device)
print("Nombre de features :", batch["features"].shape[1])


# Exercice 3
# Comparaison des optimiseurs

def train_model(
    opt_name,
    learning_rate=0.001,
    epochs=30
):

    # On utilise automatiquement le vrai nombre de features
    # Ici : 16 et non 12
    model = MLP(
        input_size=batch["features"].shape[1],
        hidden_size=128
    ).to(device)

    criterion = nn.BCELoss()


    # =================================================
    # Choix de l'optimiseur
    # =================================================

    if opt_name == "SGD":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Momentum":

        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif opt_name == "RMSprop":

        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Adam":

        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    else:

        raise ValueError(
            f"Optimiseur inconnu : {opt_name}"
        )


    # =================================================
    # TensorBoard
    # =================================================

    writer = SummaryWriter(
        f"runs/cardio_{opt_name}_lr{learning_rate}"
    )


    print("\n======================================")
    print(f"Optimiseur : {opt_name}")
    print("======================================")


    # =================================================
    # Entraînement
    # =================================================

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0


        # IMPORTANT :
        # batch_train et non batch
        for batch_train in train_loader:

            inputs = batch_train["features"].to(device)

            targets = batch_train["labels"].to(device)


            optimizer.zero_grad()


            loss = criterion(
                model(inputs),
                targets
            )


            loss.backward()


            optimizer.step()


            running_loss += loss.item()


        # Loss moyenne de l'epoch
        average_loss = (
            running_loss /
            len(train_loader)
        )


        # Affichage terminal
        print(
            f"{opt_name} - "
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f}"
        )


        # Logging TensorBoard
        writer.add_scalar(
            "Training Loss",
            average_loss,
            epoch
        )


    writer.close()
def evaluate_model(model, test_loader):

    model.eval()

    all_targets = []
    all_preds_probs = []

    # Désactiver le calcul des gradients
    with torch.no_grad():

        for batch_test in test_loader:

            inputs = batch_test["features"].to(device)
            targets = batch_test["labels"].to(device)

            outputs = model(inputs)

            # Stocker les vraies classes
            all_targets.extend(
                targets.cpu().numpy().ravel()
            )

            # Stocker les probabilités prédites
            all_preds_probs.extend(
                outputs.cpu().numpy().ravel()
            )

    # Conversion en tableaux numpy
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)

    # Passage probabilité -> classe
    # seuil = 0.5
    all_preds_classes = (
        all_preds_probs > 0.5
    ).astype(int)

    # Calcul des métriques
    precision = precision_score(
        all_targets,
        all_preds_classes
    )

    recall = recall_score(
        all_targets,
        all_preds_classes
    )

    f1 = f1_score(
        all_targets,
        all_preds_classes
    )

    auc = roc_auc_score(
        all_targets,
        all_preds_probs
    )

    print("Evaluation sur le TEST SET")

    print(
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"AUC: {auc:.4f}"
    )

# Ancien exercice 2 : L1 / L2
# On le garde commenté

"""

model = MLP(
    input_size=batch['features'].shape[1],
    hidden_size=128
).to(device)

criterion = nn.BCELoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01
)


# Première expérience :

# l1_lambda = 1e-4
# l2_lambda = 1e-3


# Deuxième expérience :

l1_lambda = 0.1
l2_lambda = 0


for epoch in range(10):

    model.train()

    running_loss = 0.0


    for batch_train in train_loader:

        inputs = batch_train["features"].to(device)

        targets = batch_train["labels"].to(device)


        optimizer.zero_grad()


        outputs = model(inputs)


        base_loss = criterion(
            outputs,
            targets
        )


        # Calcul de la pénalité L1
        l1_penalty = sum(
            p.abs().sum()
            for p in model.parameters()
        )


        # Calcul de la pénalité L2
        l2_penalty = sum(
            p.pow(2).sum()
            for p in model.parameters()
        )


        # Loss totale
        loss = (
            base_loss
            + l1_lambda * l1_penalty
            + l2_lambda * l2_penalty
        )


        loss.backward()


        optimizer.step()


        running_loss += loss.item()


    average_loss = (
        running_loss /
        len(train_loader)
    )


    print(
        f"Epoch {epoch + 1}/10 "
        f"- Loss: {average_loss:.4f}"
    )

"""


# Lancement des expériences
"""
for opt in [
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam"
]:

    train_model(
        opt,
        learning_rate=0.001
    )"""


models = {}

for opt in [
    "SGD",
    "Momentum",
    "RMSprop",
    "Adam"
]:

    models[opt] = train_model(
        opt,
        learning_rate=0.001
    )


# Evaluation du modèle Adam
evaluate_model(
    models["Adam"],
    test_loader
)