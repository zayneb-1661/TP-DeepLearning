import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import train_dataset, test_dataset
from model import ImageCaptioningModel, cnn_features
from text_utils import tokenize


def encode_captions(caption_groups, word_to_idx, device):
    # Collation Flickr30k : 5 groupes de descriptions ; chaque groupe contient B phrases.
    # On prend la première légende de chaque image (comme dans l'énoncé).
    sentences = caption_groups[0]
    ids = [tokenize(sentence, word_to_idx, max_length=20) for sentence in sentences]
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.no_grad()
def generate_caption(model, image, idx_to_word, max_new_words=19):
    """Vraie génération autoregressive : aucun mot cible n'est fourni."""
    model.eval()
    features = model.cnn(image)
    token = torch.zeros((1, 1), dtype=torch.long, device=image.device)
    states = None
    result = []
    for _ in range(max_new_words):
        embedding = model.embeddings(token)
        hidden_seq, states = model.lstm(embedding, features, init_states=states)
        logits = model.fc_out(hidden_seq[:, -1, :])
        next_idx = int(logits.argmax(dim=-1).item())
        if next_idx == 0:
            break
        if 0 <= next_idx < len(idx_to_word):
            result.append(idx_to_word[next_idx])
        token = torch.tensor([[next_idx]], dtype=torch.long, device=image.device)
    return " ".join(result) if result else "[aucun mot généré]"


def run_epoch(model, loader, criterion, word_to_idx, device,
              optimizer=None, max_batches=None, writer=None, step_offset=0):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    loss_sum = 0.0
    token_count = 0
    first_image = None
    first_caption = None
    processed = 0

    for i, (images, caption_groups) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        captions = encode_captions(caption_groups, word_to_idx, device)
        inputs = captions[:, :-1]      # mot courant
        targets = captions[:, 1:]      # mot suivant

        if first_image is None:
            first_image = images[:1].detach().clone()
            first_caption = caption_groups[0][0]

        if training:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images, inputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images, inputs)
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

        n_tokens = int((targets != 0).sum().item())
        loss_sum += float(loss.detach().item()) * n_tokens
        token_count += n_tokens
        processed += 1

        if training and writer is not None and (i + 1) % 10 == 0:
            writer.add_scalar("Loss/Train_Step", loss.item(), step_offset + i + 1)

    if processed == 0:
        raise RuntimeError("Aucun batch traité ; vérifie max_batches et le dataset.")
    return loss_sum / max(token_count, 1), first_image, first_caption, processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Lance train.py sur un noeud GPU avec l'environnement deeplearning active.")
    device = torch.device("cuda")
    torch.manual_seed(42)

    text_data = torch.load("checkpoints/text_data.pt", map_location="cpu", weights_only=True)
    word_to_idx = text_data["word_to_idx"]
    idx_to_word = text_data["idx_to_word"]
    weights = text_data["weights"]
    embedding_model = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=0)

    model = ImageCaptioningModel(cnn_model=cnn_features,
                                 embedding_model=embedding_model,
                                 lstm_hidden_dim=512).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    run_name = f"captioning_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = Path("runs") / run_name
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    writer = SummaryWriter(str(run_dir))
    best_path = Path("checkpoints") / f"best_{run_name}.pth"
    best_val_loss = float("inf")

    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Images Train: {len(train_dataset)}, Test: {len(test_dataset)}", flush=True)
    print(f"Vocabulaire: {len(idx_to_word)}, Batch size: {args.batch_size}", flush=True)
    print(f"TensorBoard: {run_dir}", flush=True)
    print(f"Checkpoint: {best_path}", flush=True)

    try:
        global_step = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _, n_train = run_epoch(
                model, train_loader, criterion, word_to_idx, device,
                optimizer=optimizer, max_batches=args.max_train_batches,
                writer=writer, step_offset=global_step)
            global_step += n_train
            val_loss, sample_image, true_caption, _ = run_epoch(
                model, test_loader, criterion, word_to_idx, device,
                max_batches=args.max_val_batches)

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)

            # Voir les phrases à l'époque 1 puis à l'époque 10 (et à la dernière).
            if epoch in (1, 10, args.epochs):
                generated = generate_caption(model, sample_image, idx_to_word)
                print(f"Epoch {epoch:02d} - Prediction : {generated}", flush=True)
                print(f"Epoch {epoch:02d} - Target     : {true_caption}", flush=True)
                writer.add_text("Examples/Prediction", generated, epoch)
                writer.add_text("Examples/Target", true_caption, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "lstm_hidden_dim": 512,
                    "run_name": run_name,
                }, best_path)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_lr, epoch)
            writer.flush()
            print(f"Epoch {epoch:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.6g}", flush=True)
    finally:
        writer.close()
    print(f"Meilleure validation : {best_val_loss:.4f}", flush=True)
    print(f"Meilleur checkpoint : {best_path}", flush=True)


if __name__ == "__main__":
    main()
