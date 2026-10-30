import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter  # Solution de secours si tensorboard absent
from tqdm import tqdm

DATA_DIR = "/mnt/ssd/irishman"
SEQ_LENGTH = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEMPERATURES = (0.2, 0.8, 2.0)
PROMPT = "X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"


def load_data(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "train.json"), "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, "validation.json"), "r", encoding="utf-8") as f:
        val_data = json.load(f)
    return train_data, val_data


def make_vocabulary(train_data, val_data):
    all_text = "".join(song["abc notation"] for song in train_data + val_data)
    unique_chars = sorted(list(set(all_text)))
    char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
    index_to_char = {idx: char for idx, char in enumerate(unique_chars)}
    return unique_chars, char_to_index, index_to_char


def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    return sequence[:max_length]


def vectorize_string(s, char_to_index):
    return [char_to_index[char] for char in s]


class MusicDataset(Dataset):
    def __init__(self, raw_data, char_to_index, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]
        self.data = [
            vectorize_string(pad_or_truncate(song["abc notation"], seq_length), char_to_index)
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=512, rnn_type="lstm"):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type doit etre 'rnn' ou 'lstm'")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        embeds = self.embedding(x)
        out, hidden_state = self.rnn(embeds, hidden_state)
        logits = self.fc(out)
        return logits, hidden_state


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item() * y.numel()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()
    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def generate_music(model, start_sequence, char_to_index, index_to_char, output_length=300, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature doit etre strictement positive")
    model.eval()
    generated_seq = start_sequence
    missing = sorted(set(start_sequence) - set(char_to_index))
    if missing:
        raise ValueError(f"L'amorce contient des caracteres absents du vocabulaire : {missing!r}")

    input_indices = [char_to_index[c] for c in start_sequence]
    input_tensor = torch.tensor(input_indices, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logits, hidden_state = model(input_tensor, hidden_state=None)

    for _ in range(output_length):
        last_logits = logits[:, -1, :] / temperature
        probs = F.softmax(last_logits, dim=-1).squeeze(0)
        next_index = torch.multinomial(probs, num_samples=1).item()
        next_char = index_to_char[next_index]
        generated_seq += next_char
        input_tensor = torch.tensor([[next_index]], dtype=torch.long, device=DEVICE)
        logits, hidden_state = model(input_tensor, hidden_state=hidden_state)
    return generated_seq


def save_generations(model, char_to_index, index_to_char, run_name, length=300):
    out_dir = Path("generations") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for temperature in TEMPERATURES:
        abc = generate_music(model, PROMPT, char_to_index, index_to_char,
                             output_length=length, temperature=temperature)
        path = out_dir / f"temperature_{str(temperature).replace('.', '_')}.abc"
        path.write_text(abc, encoding="utf-8")
        print(f"\n========== ABC, T={temperature} ==========\n{abc}\n")
        print(f"Partition enregistree : {path}", flush=True)


def train_model(args):
    train_data, val_data = load_data(args.data_dir)
    unique_chars, char_to_index, index_to_char = make_vocabulary(train_data, val_data)
    vocab_size = len(unique_chars)
    subset_size = None if args.subset_size.lower() in ("none", "all") else int(args.subset_size)

    train_dataset = MusicDataset(train_data, char_to_index, seq_length=args.seq_length, subset_size=subset_size)
    val_dataset = MusicDataset(val_data, char_to_index, seq_length=args.seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=DEVICE.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=DEVICE.type == "cuda")

    model = MusicModel(vocab_size, args.embedding_dim, args.hidden_size, args.rnn_type).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    clip_tag = f"clip{args.max_grad_norm}" if args.max_grad_norm is not None else "noclip"
    run_name = f"{args.rnn_type}_sub{subset_size}_h{args.hidden_size}_lr{args.lr}_{clip_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    for folder in ("runs_rnn", "checkpoints", "metrics", "generations"):
        Path(folder).mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs_rnn/{run_name}")
    save_path = Path("checkpoints") / f"best_{run_name}.pth"
    metrics_path = Path("metrics") / f"{run_name}.csv"
    print(f"Device: {DEVICE}, vocab_size: {vocab_size}, train: {len(train_dataset)}, val: {len(val_dataset)}", flush=True)
    print(f"Modele: {args.rnn_type}, parametres: {count_parameters(model):,}, sauvegarde: {save_path}", flush=True)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    with metrics_path.open("w", newline="", encoding="utf-8") as fout:
        csv_writer = csv.writer(fout)
        csv_writer.writerow(["epoch", "loss_train", "loss_validation", "accuracy_validation", "grad_norm_avg"])
        try:
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_train_loss, total_train_tokens = 0.0, 0
                epoch_grad_norms = []
                for x, y in tqdm(train_loader, desc=f"[{run_name}] Epoch {epoch}/{args.epochs}", leave=False):
                    x = x.to(DEVICE, non_blocking=True)
                    y = y.to(DEVICE, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits, _ = model(x)
                    loss = loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
                    loss.backward()
                    norm_limit = args.max_grad_norm if args.max_grad_norm is not None else float("inf")
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=norm_limit)
                    grad_norm_value = grad_norm.item()
                    epoch_grad_norms.append(grad_norm_value)
                    if global_step % 20 == 0:
                        writer.add_scalar("GradNorm/step", grad_norm_value, global_step)
                    optimizer.step()
                    total_train_loss += loss.item() * y.numel()
                    total_train_tokens += y.numel()
                    global_step += 1

                avg_train_loss = total_train_loss / total_train_tokens
                avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
                val_loss, val_acc = evaluate_model(model, val_loader, loss_fn, vocab_size)
                writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Accuracy/Validation", val_acc, epoch)
                writer.add_scalar("GradNorm/Epoch_Avg", avg_grad_norm, epoch)
                writer.flush()
                csv_writer.writerow([epoch, avg_train_loss, val_loss, val_acc, avg_grad_norm])
                fout.flush()
                print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | Avg GradNorm (avant clipping): {avg_grad_norm:.3f}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "vocab_chars": unique_chars,
                        "embedding_dim": args.embedding_dim,
                        "hidden_size": args.hidden_size,
                        "rnn_type": args.rnn_type,
                        "best_val_loss": best_val_loss,
                        "epoch": epoch,
                    }, save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        print(f"Early Stopping declenche a l'epoque {epoch}.", flush=True)
                        break
        finally:
            writer.close()

    print(f"Meilleur modele sauvegarde : {save_path}", flush=True)
    if args.generate_after:
        print("\nGeneration avec le meilleur checkpoint...", flush=True)
        checkpoint = load_checkpoint(save_path)
        char_to_index = {c: i for i, c in enumerate(checkpoint["vocab_chars"])}
        index_to_char = {i: c for i, c in enumerate(checkpoint["vocab_chars"])}
        gen_model = model_from_checkpoint(checkpoint)
        save_generations(gen_model, char_to_index, index_to_char, run_name, length=args.output_length)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def model_from_checkpoint(checkpoint):
    model = MusicModel(len(checkpoint["vocab_chars"]), checkpoint["embedding_dim"],
                       checkpoint["hidden_size"], checkpoint["rnn_type"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="TP4 — RNN et LSTM sur des partitions ABC")
    parser.add_argument("--action", choices=("inspect", "count-params", "train", "generate"), default="inspect")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--rnn-type", choices=("rnn", "lstm"), default="lstm")
    parser.add_argument("--subset-size", default="10000", help="10000, 50000 ou none")
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--generate-after", action="store_true")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--output-length", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)

    if args.action == "inspect":
        train_data, val_data = load_data(args.data_dir)
        unique_chars, char_to_index, _ = make_vocabulary(train_data, val_data)
        lengths = [len(song["abc notation"]) for song in train_data]
        print(f"TRAIN_COUNT={len(train_data)}")
        print(f"VAL_COUNT={len(val_data)}")
        print(f"VOCAB_SIZE={len(unique_chars)}")
        print(f"TRAIN_AVG_LENGTH={sum(lengths) / len(lengths):.2f}")
        print(f"TRAIN_MAX_LENGTH={max(lengths)}")
        print("\n--- PREMIERE PARTITION TRAIN[0] ---\n")
        print(train_data[0]["abc notation"])
        print("\n--- TEST DATASET ---")
        ds = MusicDataset(train_data[:128], char_to_index)
        dl = DataLoader(ds, batch_size=128)
        x, y = next(iter(dl))
        print(f"x.shape = {tuple(x.shape)} ; y.shape = {tuple(y.shape)} ; x.dtype = {x.dtype}")
    elif args.action == "count-params":
        for rnn_type in ("rnn", "lstm"):
            model = MusicModel(95, 256, 512, rnn_type)
            print(f"{rnn_type.upper()} : Embedding={count_parameters(model.embedding):,}; "
                  f"Recurrent={count_parameters(model.rnn):,}; FC={count_parameters(model.fc):,}; "
                  f"Total={count_parameters(model):,}")
    elif args.action == "train":
        train_model(args)
    elif args.action == "generate":
        if not args.checkpoint:
            parser.error("--checkpoint est obligatoire avec --action generate")
        ckpt = load_checkpoint(args.checkpoint)
        model = model_from_checkpoint(ckpt)
        chars = ckpt["vocab_chars"]
        save_generations(model, {c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)},
                         "generation_" + Path(args.checkpoint).stem, length=args.output_length)


if __name__ == "__main__":
    main()
