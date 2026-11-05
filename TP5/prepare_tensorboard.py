
from pathlib import Path
import shutil

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

# Run de l'entrainement complet
SOURCE = Path("runs/captioning_20261105_095836")

# Dossier des graphiques pour le rapport
DESTINATION = Path("runs/tp5_comparison")

# Lire les donnees deja enregistrees
events = event_accumulator.EventAccumulator(
    str(SOURCE),
    size_guidance={"scalars": 0}
)

events.Reload()

# Verifier les indicateurs disponibles
print("Indicateurs trouves :", events.Tags()["scalars"])

# Recreer uniquement le dossier de comparaison
if DESTINATION.exists():
    shutil.rmtree(DESTINATION)

# Deux runs avec des noms explicites
train_writer = SummaryWriter(
    str(DESTINATION / "Train")
)

val_writer = SummaryWriter(
    str(DESTINATION / "Validation")
)

# Meme nom d'indicateur pour les deux runs
TAG = "Loss/Train_vs_Validation"

# Courbe Train
for event in events.Scalars("Loss/Train"):
    train_writer.add_scalar(
        TAG,
        event.value,
        event.step
    )

# Courbe Validation
for event in events.Scalars("Loss/Validation"):
    val_writer.add_scalar(
        TAG,
        event.value,
        event.step
    )

# Courbe Learning Rate
for event in events.Scalars("LR"):
    train_writer.add_scalar(
        "LR",
        event.value,
        event.step
    )

train_writer.close()
val_writer.close()

print("\nGraphiques prepares !")
print("Dossier :", DESTINATION)
print("Runs : Train et Validation")
print("Graphiques : Loss/Train_vs_Validation et LR")
