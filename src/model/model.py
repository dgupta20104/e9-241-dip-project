"""The recognition network: line image -> LaTeX tokens.

Design: Claude Chat Memory/CNN_BiLSTM_Attention_Model.md

    line image (1 x 64 x W)
      -> CNN                       -> 256 x 8 x W/8    (a grid of feature vectors)
      -> row BiLSTM (along W')     -> 256 x 8 x W/8    (each vector now knows its left/right)
      -> column BiLSTM (along H')  -> 256 x 8 x W/8    (... and what is above/below) = G
      -> 2-D attention + LSTM decoder, one token per step until <end>

This file also holds the input preparation (`prepare_image`) that training
AND prediction both use, so the model always sees images made the same way:

    any image -> binary mask (ink = 255) -> crop to the ink -> resize so the ink
    is 56 px high -> 4 px empty margin on every side (height 64) -> width rounded
    up to a multiple of 8 -> float tensor (1, 64, W), ink = 1.0, paper = 0.0

Shapes used in the comments:
    B = batch size, W = image width, H' = 8 and W' = W/8 = size of the CNN output grid,
    N = H' * W' = number of grid positions, T = number of decoder steps, V = vocab size.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.model.vocab import END, PAD, START

# ---- input size -----------------------------------------------------------------
IMG_H = 64          # height of every model input (px)
MARGIN = 4          # empty paper border around the ink (px)
INK_H = IMG_H - 2 * MARGIN   # height of the ink inside the input = 56 px
DOWN = 8            # the CNN makes height and width 8x smaller (3 max-pools of 2)
MAX_W = 1024        # widest input; wider expressions are squeezed sideways
MIN_W = 16          # narrowest input, so even a lone "1" gives 2 feature columns


# ---- input preparation (shared by train.py and predict.py) -------------------------
def crop_to_ink(mask):
    """Cut a binary mask (ink = 255) down to the smallest box that holds all ink.

    An empty mask (no ink at all) becomes a single paper pixel.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros((1, 1), np.uint8)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def ink_mask(img, ink_white=None):
    """Binary image -> mask with ink = 255, paper = 0, cropped to the ink.

    The input is already binary (no thresholding here), only its colours may
    be swapped:
    ink_white = True : ink is 255 on 0 (pipeline binaries) -> used as it is
    ink_white = False: ink is 0 on 255 (CROHME renders)    -> colours swapped
    ink_white = None : guess - paper covers most of the image, so if most
                       pixels are bright the ink must be dark.
    """
    if img.ndim == 3:                                     # colour -> grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ink_white is None:
        ink_white = np.median(img) < 128                  # mostly dark -> bright ink
    mask = img if ink_white else 255 - img                # make ink = 255
    return crop_to_ink(mask.astype(np.uint8))


def to_model_input(mask):
    """Cropped binary mask (ink = 255) -> float tensor (1, 64, W) with ink = 1.0."""
    h, w = mask.shape
    # Keep the shape (aspect ratio): the ink becomes INK_H px high, the width follows.
    new_w = int(round(w * INK_H / h))
    new_w = min(max(new_w, MIN_W - 2 * MARGIN), MAX_W - 2 * MARGIN)
    # INTER_AREA is the best way to shrink; INTER_LINEAR to enlarge.
    interp = cv2.INTER_AREA if h > INK_H else cv2.INTER_LINEAR
    small = cv2.resize(mask, (new_w, INK_H), interpolation=interp)

    # Paper margin on every side, plus extra paper on the right so that the
    # width is a multiple of 8 (then every CNN output column is pure image).
    total_w = new_w + 2 * MARGIN
    extra = (-total_w) % DOWN
    small = cv2.copyMakeBorder(small, MARGIN, MARGIN, MARGIN, MARGIN + extra,
                               cv2.BORDER_CONSTANT, value=0)
    # uint8 0..255 -> float 0..1, and add the channel axis: (64, W) -> (1, 64, W)
    return torch.from_numpy(small.astype(np.float32) / 255.0).unsqueeze(0)


def prepare_image(img, ink_white=None, augment=None):
    """Image (numpy, any size) -> model input tensor (1, 64, W).

    `augment` (training only) is a function mask -> mask that randomly
    changes the cropped binary mask before it is resized.
    """
    mask = ink_mask(img, ink_white)
    if augment is not None:
        mask = augment(mask)
    return to_model_input(mask)


def pad_images(images):
    """List of (1, 64, W_k) tensors -> one batch tensor (B, 1, 64, W_max) + widths.

    Narrower images are filled up on the right with paper (0). `widths` tells
    the model where each real image ends, so it can ignore the filler.
    """
    widths = torch.tensor([im.shape[-1] for im in images], dtype=torch.long)
    batch = torch.zeros(len(images), 1, IMG_H, int(widths.max()))
    for k, im in enumerate(images):
        batch[k, :, :, :im.shape[-1]] = im
    return batch, widths


def get_device(name=None, verbose=False):
    """Pick where the maths runs: NVIDIA GPU (cuda) > Apple GPU (mps) > CPU.

    `name` forces a device ("cuda", "mps", "cpu"). With verbose=True it prints
    which one is used.
    """
    if name:
        device = torch.device(name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if verbose:
        if device.type == "cuda":
            print(f"Device: GPU (CUDA) - {torch.cuda.get_device_name(device)}")
        elif device.type == "mps":
            print("Device: GPU (Apple MPS)")
        else:
            print("Device: CPU - no GPU in use, training will be very slow")
    return device


# ---- 1. CNN: image -> grid of feature vectors ---------------------------------------
def conv_bn_relu(c_in, c_out):
    """3x3 convolution (keeps the size) + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),  # BatchNorm has its own bias
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """(B, 1, 64, W) -> (B, 256, 8, W/8). Channels double while the size halves.

    Receptive field: one output vector "sees" a 62 x 62 px patch of the input,
    about one symbol plus its neighbours.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            conv_bn_relu(1, 32), nn.MaxPool2d(2),        # block 1 -> 32 x 32 x W/2
            conv_bn_relu(32, 64), nn.MaxPool2d(2),       # block 2 -> 64 x 16 x W/4
            conv_bn_relu(64, 128), conv_bn_relu(128, 128),
            nn.MaxPool2d(2),                             # block 3 -> 128 x 8 x W/8
            conv_bn_relu(128, 256), conv_bn_relu(256, 256),   # block 4 -> 256 x 8 x W/8
        )

    def forward(self, x):
        return self.layers(x)


# ---- 2. Encoder: CNN + row BiLSTM + column BiLSTM -------------------------------------
class Encoder(nn.Module):
    """Image batch -> G (B, N, 256): one context-aware vector per grid position.

    Row BiLSTM: every grid row is read left->right and right->left, so each
    vector learns what is to its left and right. Column BiLSTM: every grid
    column is then read top->bottom and bottom->top, so each vector also
    learns what is above and below it (needed for fractions, powers, ...).
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.cnn = CNN()
        # bidirectional=True -> output = forward half + backward half = 2 * hidden = 256
        self.row_lstm = nn.LSTM(256, hidden, batch_first=True, bidirectional=True)
        self.col_lstm = nn.LSTM(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.out_dim = 2 * hidden

    def forward(self, images, widths):
        """images (B, 1, 64, W), widths (B,) real image widths in px.

        Returns G (B, N, 256), mask (B, N) True where the position is real
        image (False = filler on the right), and the grid size (H', W').
        """
        feat = self.cnn(images)                                  # (B, C, H', W')
        B, C, Hp, Wp = feat.shape
        # How many grid columns of each image are real (not filler).
        feat_w = (widths // DOWN).clamp(1, Wp)                   # (B,)

        # --- row BiLSTM: every row of every image is one sequence of W' vectors
        x = feat.permute(0, 2, 3, 1).reshape(B * Hp, Wp, C)     # (B*H', W', C)
        row_len = feat_w.repeat_interleave(Hp)                   # length of each row sequence
        # "Packing" tells the LSTM where each row really ends, so the
        # right->left pass starts at the real end and never reads the filler.
        packed = pack_padded_sequence(x, row_len.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.row_lstm(packed)
        x, _ = pad_packed_sequence(out, batch_first=True, total_length=Wp)  # (B*H', W', 2d)

        # --- column BiLSTM: every column is one sequence of H' vectors
        x = x.reshape(B, Hp, Wp, -1).permute(0, 2, 1, 3)         # (B, W', H', 2d)
        x = x.reshape(B * Wp, Hp, -1)                            # (B*W', H', 2d)
        x, _ = self.col_lstm(x)                                  # (B*W', H', 2d)

        # --- back to one list of N = H' * W' vectors per image (row by row)
        G = x.reshape(B, Wp, Hp, -1).permute(0, 2, 1, 3)         # (B, H', W', 2d)
        G = G.reshape(B, Hp * Wp, -1)                            # (B, N, 2d)

        # mask[b, n] = True if position n of image b lies inside the real image
        col_ok = torch.arange(Wp, device=images.device)[None, :] < feat_w[:, None]  # (B, W')
        mask = col_ok[:, None, :].expand(B, Hp, Wp).reshape(B, Hp * Wp)
        return G, mask, (Hp, Wp)


# ---- 3. 2-D attention: where should the decoder look now? ----------------------------
class Attention(nn.Module):
    """Additive attention over all N grid positions.

        score  e_n = w^T tanh(W_s s + W_g g_n [+ W_beta beta_n])
        weight a_n = softmax(e)_n         (all weights >= 0, they sum to 1)
        context c  = sum_n a_n g_n        (weighted average of the grid vectors)

    beta_n (coverage, optional) = attention already spent on position n in
    earlier steps; it helps the model not to read a symbol twice or skip one.
    """

    def __init__(self, enc_dim=256, dec_dim=256, att_dim=256, use_coverage=False):
        super().__init__()
        self.W_s = nn.Linear(dec_dim, att_dim, bias=False)   # decoder memory -> att space
        self.W_g = nn.Linear(enc_dim, att_dim)               # grid vector    -> att space
        self.w = nn.Linear(att_dim, 1, bias=False)           # att space      -> one score
        self.W_beta = nn.Linear(1, att_dim, bias=False) if use_coverage else None

    def forward(self, s, G, G_proj, mask, coverage=None):
        """s (B, n) decoder memory; G (B, N, enc_dim); G_proj = W_g G (B, N, a),
        computed once per image because it does not change between steps;
        mask (B, N); coverage (B, N) or None. Returns context (B, enc_dim), alpha (B, N)."""
        z = G_proj + self.W_s(s).unsqueeze(1)                    # (B, N, a)
        if self.W_beta is not None and coverage is not None:
            z = z + self.W_beta(coverage.unsqueeze(-1))
        e = self.w(torch.tanh(z)).squeeze(-1)                    # (B, N) one score per position
        e = e.masked_fill(~mask, float("-inf"))                  # never look at the filler
        alpha = torch.softmax(e, dim=1)                          # (B, N), sums to 1
        context = torch.bmm(alpha.unsqueeze(1), G).squeeze(1)    # (B, enc_dim)
        return context, alpha


# ---- 4. The full model: encoder + attention + LSTM decoder ------------------------------
class Im2LatexModel(nn.Module):
    """Line image -> sequence of LaTeX token indices.

    Decoder, one step i (one output token):
        s_i   = LSTMCell([E(y_{i-1}); c_{i-1}], s_{i-1})    read previous token + previous context
        c_i   = Attention(s_i, G)                           look at the image
        p_i   = softmax(W_o tanh(W_c [s_i; c_i]))           probabilities of the V tokens
    Start: s_0 = tanh(W_0 * mean(G)), c_0 = 0, y_0 = <start>.
    """

    def __init__(self, vocab_size, use_coverage=False, dropout=0.0,
                 enc_hidden=128, emb_dim=256, dec_dim=256, att_dim=256):
        super().__init__()
        # Saved inside every checkpoint so predict.py can rebuild the same model.
        self.config = dict(vocab_size=vocab_size, use_coverage=use_coverage, dropout=dropout,
                           enc_hidden=enc_hidden, emb_dim=emb_dim, dec_dim=dec_dim,
                           att_dim=att_dim)
        self.use_coverage = use_coverage

        self.encoder = Encoder(enc_hidden)
        enc_dim = self.encoder.out_dim                                  # 256

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)  # E(y): token -> vector
        self.init_s = nn.Linear(enc_dim, dec_dim)                       # W_0
        self.cell = nn.LSTMCell(emb_dim + enc_dim, dec_dim)             # decoder memory update
        self.attention = Attention(enc_dim, dec_dim, att_dim, use_coverage)
        self.W_c = nn.Linear(dec_dim + enc_dim, dec_dim)
        self.dropout = nn.Dropout(dropout)                              # 0 = off (see tuning guide)
        self.W_o = nn.Linear(dec_dim, vocab_size)                       # -> one score per token

    # -- helpers ------------------------------------------------------------------------
    def _start_state(self, G, mask):
        """Initial decoder state from the image: s_0 = tanh(W_0 * mean of the real G vectors)."""
        m = mask.unsqueeze(-1).float()                               # (B, N, 1)
        g_mean = (G * m).sum(1) / m.sum(1).clamp(min=1)              # (B, enc_dim)
        s = torch.tanh(self.init_s(g_mean))                          # hidden state s_0
        cell_mem = torch.zeros_like(s)                               # LSTM's internal cell memory
        context = torch.zeros_like(g_mean)                           # c_0 = 0
        coverage = torch.zeros(mask.shape, device=G.device) if self.use_coverage else None
        return s, cell_mem, context, coverage

    def _step(self, y_prev, s, cell_mem, context, coverage, G, G_proj, mask):
        """One decoder step -> (logits (B, V), new state..., alpha (B, N))."""
        x = torch.cat([self.embed(y_prev), context], dim=1)          # [E(y_{i-1}); c_{i-1}]
        s, cell_mem = self.cell(x, (s, cell_mem))                    # s_i
        context, alpha = self.attention(s, G, G_proj, mask, coverage)   # c_i
        if coverage is not None:
            coverage = coverage + alpha                              # attention spent so far
        hidden = torch.tanh(self.W_c(torch.cat([s, context], dim=1)))
        logits = self.W_o(self.dropout(hidden))                      # softmax is applied by the loss
        return logits, s, cell_mem, context, coverage, alpha

    # -- training: teacher forcing --------------------------------------------------------
    def forward(self, images, widths, tokens_in):
        """images (B, 1, 64, W), widths (B,), tokens_in (B, T) = <start> y_1 .. y_{T-1}.

        Teacher forcing: at every step the decoder gets the TRUE previous
        token (not its own guess), so all steps can be trained at once.
        Returns logits (B, T, V): scores for the token at each step.
        """
        G, mask, _ = self.encoder(images, widths)
        G_proj = self.attention.W_g(G)                               # same for every step
        s, cell_mem, context, coverage = self._start_state(G, mask)

        all_logits = []
        for t in range(tokens_in.shape[1]):
            logits, s, cell_mem, context, coverage, _ = self._step(
                tokens_in[:, t], s, cell_mem, context, coverage, G, G_proj, mask)
            all_logits.append(logits)
        return torch.stack(all_logits, dim=1)                        # (B, T, V)

    # -- prediction: greedy decoding ---------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, images, widths, max_len=150, return_attention=False):
        """Read a batch of images; at every step take the most likely token and
        feed it back in as the next input, until every image has produced <end>
        (or max_len steps).

        Returns token indices (B, T) (everything after <end> is junk, cut it with
        Vocab.decode) and, with return_attention=True, also the attention maps
        (B, T, H', W'): where the model looked for each token (report heat maps).
        """
        G, mask, (Hp, Wp) = self.encoder(images, widths)
        G_proj = self.attention.W_g(G)
        s, cell_mem, context, coverage = self._start_state(G, mask)

        B = images.shape[0]
        y = torch.full((B,), START, dtype=torch.long, device=images.device)
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        outputs, maps = [], []
        for _ in range(max_len):
            logits, s, cell_mem, context, coverage, alpha = self._step(
                y, s, cell_mem, context, coverage, G, G_proj, mask)
            y = logits.argmax(dim=1)                                 # most likely token
            outputs.append(y)
            if return_attention:
                maps.append(alpha.reshape(B, Hp, Wp))
            finished |= y == END
            if finished.all():                                       # everyone is done
                break

        tokens = torch.stack(outputs, dim=1)                         # (B, T)
        if return_attention:
            return tokens, torch.stack(maps, dim=1)                  # (B, T, H', W')
        return tokens


def count_parameters(model):
    """Number of trainable numbers (weights) in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
