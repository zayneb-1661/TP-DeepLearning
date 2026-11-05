
import os
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# TP5 - Question 1.b : Embeddings Word2Vec
# ============================================================

CAPTIONS_FILE = "/mnt/ssd/flickr30k/results_20130124.token"
WORD2VEC_FILE = "/mnt/ssd/word2vec/GoogleNews-vectors-negative300.bin"

RANDOMIZED = False
EMBEDDING_DIM = 300
MAX_LENGTH = 20

torch.manual_seed(42)

# ============================================================
# 1. Construction du vocabulaire
# ============================================================

print("Construction du vocabulaire...")

vocabulary = set()

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", 1)

        if len(parts) != 2:
            continue

        _, sentence = parts

        for word in sentence.lower().split():
            vocabulary.add(word)

vocabulary = sorted(vocabulary)

print("Mots uniques :", len(vocabulary))


# ============================================================
# 2. Lecture des vecteurs Word2Vec
# ============================================================

def read_word2vec():
    with open(WORD2VEC_FILE, "rb") as f:

        header = f.readline()
        n_words, n_dim = map(int, header.split())

        for _ in range(n_words):
            chars = bytearray()

            while True:
                char = f.read(1)

                if char == b" ":
                    break

                if not char:
                    raise EOFError("Fichier Word2Vec incomplet")

                chars.extend(char)

            # Nettoyage du mot avant comparaison
            word = chars.decode(
                "utf-8", errors="ignore"
            ).strip()

            raw = f.read(n_dim * 4)

            if len(raw) != n_dim * 4:
                raise EOFError("Vecteur Word2Vec incomplet")

            vector = np.frombuffer(
                raw, dtype="<f4"
            )

            yield word, vector


# ============================================================
# 3. Creation de la couche Embedding
# ============================================================

def create_embedding_layer(vocab, randomized=False):

    # Indices reserves
    # 0 : Padding
    # 1 : Mot inconnu

    idx_to_word = ["<PAD>", "<UNK>"] + list(vocab)

    word_to_idx = {
        word: idx
        for idx, word in enumerate(idx_to_word)
    }

    # Initialisation des embeddings
    weights = torch.randn(
        len(idx_to_word),
        EMBEDDING_DIM
    ) * 0.02

    # Le padding est represente par un vecteur nul
    weights[0].zero_()

    # Ensemble pour compter les mots distincts trouves
    found_indices = set()

    if not randomized:

        print("Chargement des vecteurs Word2Vec...")

        for word, vector in read_word2vec():

            key = word.lower().strip()

            if key in word_to_idx:

                idx = word_to_idx[key]

                # Eviter les doublons et les tokens speciaux
                if idx >= 2 and idx not in found_indices:

                    weights[idx] = torch.from_numpy(
                        vector.copy()
                    )

                    found_indices.add(idx)

        print(
            "Mots uniques couverts par Word2Vec :",
            len(found_indices)
        )

        coverage = (
            len(found_indices) / len(vocab) * 100
            if len(vocab) > 0 else 0
        )

        print(
            f"Taux de couverture : {coverage:.2f}%"
        )

    # Creation de la couche PyTorch
    emb_layer = nn.Embedding.from_pretrained(
        weights,
        freeze=not randomized,
        padding_idx=0
    )

    return emb_layer, idx_to_word, word_to_idx


# ============================================================
# 4. Tokenisation des descriptions
# ============================================================

def tokenize(sentence, word2idx, max_length=MAX_LENGTH):

    # Premier indice reserve, comme dans le code precedent
    tokens = [0]

    sentence = sentence.lower()

    for word in sentence.split():

        idx = word2idx.get(word, 1)

        tokens.append(idx)

    # Tronquer si la sequence est trop longue
    tokens = tokens[:max_length]

    # Padding si la sequence est trop courte
    tokens += [0] * (max_length - len(tokens))

    return tokens


# ============================================================
# 5. Conversion des indices en mots
# ============================================================

def untokenize(tokens, idx2word):

    words = []

    for index in tokens:

        index = int(index)

        if 0 < index < len(idx2word):
            words.append(idx2word[index])

    return words


# ============================================================
# 6. Verification et sauvegarde
# ============================================================

if __name__ == "__main__":

    embedding, idx2word, word2idx = create_embedding_layer(
        vocabulary,
        randomized=RANDOMIZED
    )

    print("\n===== VERIFICATION =====")

    print(
        "Dimensions embeddings :",
        tuple(embedding.weight.shape)
    )

    print(
        "Parametres entrainables :",
        embedding.weight.requires_grad
    )

    sentence = "A man is walking"

    tokens = tokenize(sentence, word2idx)

    decoded = " ".join(
        untokenize(tokens, idx2word)
    )

    print("Exemple :", sentence)
    print("Tokens :", tokens)
    print("Decodage :", decoded)

    # Sauvegarde
    os.makedirs("checkpoints", exist_ok=True)

    save_path = "checkpoints/text_data.pt"

    torch.save({
        "weights": embedding.weight.detach().cpu(),
        "word_to_idx": word2idx,
        "idx_to_word": idx2word,
        "freeze": not RANDOMIZED
    }, save_path)

    print("Fichier sauvegarde :", save_path)
