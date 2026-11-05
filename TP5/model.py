
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math

# ==================================================
# EXERCICE 2.a : ResNet50 pre-entraine
# ==================================================

# 1. Charger ResNet50 avec les poids ImageNet
resnet_pretrained = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# 2. Retirer AdaptiveAvgPool2d et Linear
cnn_features = nn.Sequential(
    *list(resnet_pretrained.children())[:-2]
)

# 3. Geler les poids du ResNet
for param in cnn_features.parameters():
    param.requires_grad = False

# ==================================================
# EXERCICE 3.a : Attention spatiale
# ==================================================

class AttentionModule(nn.Module):

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.attention_layer = nn.Conv2d(
            feature_dim + hidden_dim,
            1,
            kernel_size=1
        )

    def forward(self, features, hidden_state):

        # features : (batch_size, 2048, 7, 7)
        # hidden_state : (batch_size, hidden_dim)

        batch_size, _, H, W = features.shape

        # 1. Etaler l'etat cache sur les 49 regions
        hidden_state_expanded = (
            hidden_state.unsqueeze(2)
                        .unsqueeze(3)
                        .repeat(1, 1, H, W)
        )

        # 2. Concatenation sur les canaux
        concat_features = torch.cat(
            (features, hidden_state_expanded),
            dim=1
        )

        # 3. Calcul des scores d'attention
        attention_scores = (
            self.attention_layer(concat_features)
                .squeeze(1)
        )

        # 4. Softmax sur les 49 regions
        attention_weights = F.softmax(
            attention_scores.reshape(batch_size, -1),
            dim=1
        )

        attention_weights = attention_weights.reshape(
            batch_size, 1, H, W
        )

        # 5. Somme ponderee sur H et W
        context_vector = torch.sum(
            features * attention_weights,
            dim=(2, 3)
        )

        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):

    def __init__(self, input_size, hidden_size,
                 attention_module, attention_size):

        super().__init__()

        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Mot + etat cache precedent + contexte visuel
        total_dim = input_size + hidden_size + attention_size

        # Les quatre portes du LSTM
        self.lin_i = nn.Linear(total_dim, hidden_size)
        self.lin_f = nn.Linear(total_dim, hidden_size)
        self.lin_c = nn.Linear(total_dim, hidden_size)
        self.lin_o = nn.Linear(total_dim, hidden_size)

        self.init_weights()

    def init_weights(self):

        stdv = 1.0 / math.sqrt(self.hidden_size)

        # Initialiser uniquement les portes du LSTM
        for layer in [
            self.lin_i, self.lin_f,
            self.lin_c, self.lin_o
        ]:
            nn.init.uniform_(layer.weight, -stdv, stdv)
            nn.init.uniform_(layer.bias, -stdv, stdv)

    def forward(self, x, features, init_states=None):

        # x : (batch_size, sequence_size, input_size)
        # features : (batch_size, 2048, 7, 7)

        bs, seq_sz, _ = x.size()
        hidden_seq = []

        # 1. Initialisation des etats
        if init_states is None:
            h_t = x.new_zeros(bs, self.hidden_size)
            c_t = x.new_zeros(bs, self.hidden_size)
        else:
            h_t, c_t = init_states

        # 2. Parcours de la sequence mot par mot
        for t in range(seq_sz):

            x_t = x[:, t, :]

            # 3. Attention sur les 49 regions
            att_t, _ = self.attention_module(
                features, h_t
            )

            # 4. Concatenation
            concat = torch.cat(
                (att_t, x_t, h_t),
                dim=1
            )

            # 5. Calcul des quatre portes
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # 6. Mise a jour de la memoire
            c_t = f_t * c_t + i_t * g_t

            # 7. Nouvel etat cache
            h_t = o_t * torch.tanh(c_t)

            # 8. Conserver l'etat cache du pas courant
            hidden_seq.append(h_t.unsqueeze(1))

        # Regrouper tous les etats caches
        hidden_seq = torch.cat(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)


# ==================================================
# EXERCICE 4.a : Modele global Image Captioning
# ==================================================

class ImageCaptioningModel(nn.Module):

    def __init__(self, cnn_model, embedding_model,
                 lstm_hidden_dim=512):

        super().__init__()

        # 1. ResNet50 sans les deux dernieres couches
        # cnn_model est deja notre extracteur cnn_features
        self.cnn = cnn_model

        # Geler les poids de ResNet
        self.cnn.requires_grad_(False)
        self.cnn.eval()

        # 2. Module d'attention
        self.attention = AttentionModule(
            feature_dim=2048,
            hidden_dim=lstm_hidden_dim
        )

        # 3. Embeddings Word2Vec
        self.embeddings = embedding_model

        vocab_size = embedding_model.weight.shape[0]
        emb_dim = embedding_model.weight.shape[1]

        # 4. LSTM avec attention
        self.lstm = LSTMWithAttention(
            input_size=emb_dim,
            hidden_size=lstm_hidden_dim,
            attention_module=self.attention,
            attention_size=2048
        )

        # 5. Projection vers le vocabulaire
        self.fc_out = nn.Linear(
            lstm_hidden_dim,
            vocab_size
        )

    def forward(self, images, captions):

        # ResNet reste en mode evaluation
        # car ses poids sont geles
        self.cnn.eval()

        with torch.no_grad():
            cnn_features = self.cnn(images)

        # Conversion des indices en vecteurs Word2Vec
        emb = self.embeddings(captions)

        # LSTM + attention
        lstm_outputs, (hidden_state, cell_state) = (
            self.lstm(emb, cnn_features)
        )

        # Scores de prediction des mots
        output = self.fc_out(lstm_outputs)

        return output

# ==================================================
# Verification
# ==================================================

if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    cnn_features = cnn_features.to(device)
    cnn_features.eval()

    # Test avec deux images
    images = torch.randn(2, 3, 224, 224).to(device)

    with torch.no_grad():
        outputs = cnn_features(images)

    print("Device :", device)
    print("Entree :", images.shape)
    print("Sortie :", outputs.shape)

    trainable_params = sum(
        p.numel()
        for p in cnn_features.parameters()
        if p.requires_grad
    )

    print("Parametres entrainables :", trainable_params)
    
    print("\n===== TEST ATTENTION =====")

    attention = AttentionModule(
        feature_dim=2048,
        hidden_dim=512
    ).to(device)

    hidden_state = torch.randn(2, 512).to(device)

    context_vector, attention_weights = attention(
        outputs,
        hidden_state
    )

    print("Features :", outputs.shape)
    print("Hidden state :", hidden_state.shape)
    print("Context vector :", context_vector.shape)
    print("Attention weights :", attention_weights.shape)

    print(
        "Somme des poids d'attention :",
        attention_weights.sum(dim=(1, 2, 3))
    )
    print("\n===== TEST LSTM AVEC ATTENTION =====")

    lstm = LSTMWithAttention(
        input_size=300,
        hidden_size=512,
        attention_module=attention,
        attention_size=2048
    ).to(device)

    # Simulation de 2 descriptions de 5 mots
    x = torch.randn(2, 5, 300).to(device)

    hidden_seq, (h_t, c_t) = lstm(x, outputs)

    print("Entree Word2Vec :", x.shape)
    print("Features CNN :", outputs.shape)
    print("Sequence cachee :", hidden_seq.shape)
    print("Dernier etat h_t :", h_t.shape)
    print("Derniere memoire c_t :", c_t.shape)

    print("\n===== TEST MODELE GLOBAL =====")

    # 1. Charger les embeddings sauvegardes
    text_data = torch.load(
        "checkpoints/text_data.pt",
        map_location="cpu",
        weights_only=True
    )

    embedding_model = nn.Embedding.from_pretrained(
        text_data["weights"],
        freeze=text_data["freeze"],
        padding_idx=0
    )

    # 2. Construire le modele global
    captioning_model = ImageCaptioningModel(
        cnn_model=cnn_features,
        embedding_model=embedding_model,
        lstm_hidden_dim=512
    ).to(device)

    captioning_model.eval()

    # 3. Simuler deux descriptions de cinq tokens
    vocab_size = len(text_data["idx_to_word"])

    captions = torch.randint(
        low=2,
        high=vocab_size,
        size=(2, 5),
        device=device
    )

    # 0 = position initiale reservee
    captions[:, 0] = 0

    # 4. Tester une prediction
    with torch.no_grad():
        predictions = captioning_model(images, captions)

    print("Images :", images.shape)
    print("Captions :", captions.shape)
    print("Predictions :", predictions.shape)

    print(
        "Taille du vocabulaire :",
        vocab_size
    )
