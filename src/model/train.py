"""Train the recognition model on the rendered CROHME 2019 images.

Needs (made earlier):
    data/processed/crohme/<split>_truth.tsv + images   (python -m src.model.data_prep)
    data/processed/crohme/vocab.txt                    (python -m src.model.vocab)

What happens:
    1. print which device is used (GPU or CPU)
    2. every epoch, train and valid are measured THE SAME WAY, so they can be
       compared (train much better than valid = overfitting, both bad = underfitting):
         - train on all train images (teacher forcing, random augmentation)
           -> train_loss
         - valid loss: same cross-entropy, teacher forcing, no augmentation
           -> valid_loss
         - greedy decoding (the model uses its own guesses) on FIXED random subsets
           of the valid set (--valid-eval-size images, default 300) and of the
           train set (--train-eval-size images), chosen once, no augmentation.
           (Decoding is the slow part, so not the whole valid set every epoch.) ->
             ExpRate     = % of expressions that are 100 % correct (+ with <= 1 / <= 2 token errors)
             CER         = token edit distance / number of true tokens (lower is better)
             token acc.  = 1 - CER, in %
         - learning rate / best weights / early stopping use the valid-subset CER only
         - the numbers and the best weights are only kept in memory
       The TEST set is never looked at during training.
    3. after training (also after early stopping or Ctrl+C): put back the best
       weights, measure the FULL valid set and the test set ONCE, and only THEN
       write everything

Written to outputs/model/ once, at the end of training:
    best.pt                 weights of the epoch with the lowest valid CER (+ vocab, config)
    history.csv             one row per epoch: epoch, lr, train_loss, valid_loss,
                            train_* and valid_* (exprate, le1, le2, cer, token_acc), seconds
    accuracy_vs_epoch.png   train vs valid: loss | ExpRate | token accuracy
    test_results.json       model config (channels, dropouts) + train-subset, full valid
                            and test numbers of the best checkpoint
    test_predictions.tsv    image, truth, prediction, edit distance (test, best checkpoint)

Against overfitting (tuning guide in CNN_BiLSTM_Attention_Model.md):
    --enc-dropout p    dropout in the encoder (row-BiLSTM output and final grid G)
    --dropout p        dropout in the decoder (before the output layer)
    augmentation       stroke width, slant, stretch, rotation, elastic warp, blur, specks

Usage (from the project root):
    python -m src.model.train                                  # full training
    python -m src.model.train --enc-dropout 0.5 --dropout 0.5  # with both dropouts
    python -m src.model.train --limit 64 --epochs 2 --batch 8 --train-eval-size 32  # quick test
"""

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:          # allow `python src/model/train.py` / IDE Run button
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")                  # draw plots into files, no window needed
import matplotlib.pyplot as plt  # noqa: E402

from src.model.model import (CNN_CHANNELS, Im2LatexModel, count_parameters,  # noqa: E402
                             crop_to_ink, get_device, pad_images, prepare_image)
from src.model.vocab import DATA, PAD, VOCAB_PATH, Vocab, read_truth_tsv, tokenize  # noqa: E402

OUT = ROOT / "outputs" / "model"
MAX_LEN = 150          # longest label the decoder has to write (train max is 129)


# ---- data augmentation (train only) ----------------------------------------------------
# Every epoch each training image gets a NEW random variation, so the model sees
# many slightly different versions and learns the symbols, not the exact pixels.
# Nothing is saved to disk.
# Sizes below are in render pixels: a typical CROHME symbol is ~48 px high,
# strokes are 2-4 px wide (src/model/data_prep.py).
def shear(mask, k):
    """Slant the writing: every row moves sideways by k * (its row number).

    |k| = 0.3 is a lean of about 17 degrees. The canvas grows so no ink is cut off.
    """
    h, w = mask.shape
    shift = max(0.0, -k * (h - 1))              # when rows move left, push all back right
    new_w = w + int(np.ceil(abs(k) * (h - 1))) + 1
    M = np.float32([[1, k, shift], [0, 1, 0]])  # x' = x + k*y + shift, y' = y
    return cv2.warpAffine(mask, M, (new_w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def elastic_warp(mask, step=32, amount=4.0):
    """Bend the strokes a little, as if another hand wrote the same thing.

    A coarse grid (one point every `step` px) gets random shifts of up to
    `amount` px; smooth (cubic) interpolation spreads them to every pixel, so
    the strokes wobble gently but stay connected.
    """
    pad = int(np.ceil(amount)) + 1              # room so no ink is pushed off the edge
    mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    h, w = mask.shape
    gh, gw = h // step + 2, w // step + 2       # size of the coarse grid
    # torch random numbers, not numpy's (see augment_tensor)
    shifts = (torch.rand(2, gh, gw) * 2 - 1).numpy() * amount
    dx = cv2.resize(shifts[0], (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(shifts[1], (w, h), interpolation=cv2.INTER_CUBIC)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    # output pixel (x, y) takes the input pixel at (x + dx, y + dy)
    return cv2.remap(mask, xs + dx, ys + dy, cv2.INTER_LINEAR, borderValue=0)


def augment_mask(mask):
    """Random shape changes on the cropped binary mask (ink = 255), before resizing."""
    # 1) stroke width: thicker (dilate) or thinner (erode) pen. Thicker is more
    #    common, because photo crops from the pipeline have thicker strokes than
    #    the renders.
    r = random.random()
    if r < 0.4:
        k = random.choice([2, 3, 4])            # strokes get 1 to 3 px wider
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    elif r < 0.6:
        k = random.choice([2, 3])               # strokes get 1 to 2 px thinner
        thinner = cv2.erode(mask, np.ones((k, k), np.uint8))
        # only keep it if most of the ink survives (thin strokes could vanish)
        if (thinner > 127).sum() >= 0.4 * (mask > 127).sum():
            mask = thinner

    # 2) slant: writing leaning left or right, up to about 17 degrees
    if random.random() < 0.5:
        mask = shear(mask, random.uniform(-0.3, 0.3))

    # 3) sideways stretch / squeeze (handwriting width varies)
    if random.random() < 0.5:
        h, w = mask.shape
        new_w = max(1, int(round(w * random.uniform(0.8, 1.2))))
        mask = cv2.resize(mask, (new_w, h), interpolation=cv2.INTER_NEAREST)

    # 4) small rotation, -3 to +3 degrees (Step 2 never levels a line perfectly)
    if random.random() < 0.5:
        h, w = mask.shape
        angle = random.uniform(-3.0, 3.0)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # make the canvas big enough that no ink is cut off
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w, new_h = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
        M[0, 2] += new_w / 2 - w / 2
        M[1, 2] += new_h / 2 - h / 2
        mask = cv2.warpAffine(mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

    # 5) elastic warp: strokes bend a little (shifts of up to 4 px, ~8 % of a symbol)
    if random.random() < 0.5:
        mask = elastic_warp(mask)

    return crop_to_ink(mask)                    # the changes may move the ink box


def augment_tensor(x):
    """Random blur / specks on the final (1, 64, W) input (values 0..1)."""
    # 6) blur: softer stroke edges
    if random.random() < 0.3:
        img = cv2.GaussianBlur(x[0].numpy(), (3, 3), random.uniform(0.5, 1.0))
        x = torch.from_numpy(img).unsqueeze(0)
    # 7) noise: a few random pixels flipped, like leftover specks after thresholding.
    #    (torch's random numbers, not numpy's: each image-loading worker gets its
    #    own torch seed, numpy would give every worker the same "random" specks)
    if random.random() < 0.3:
        flip = torch.rand(x.shape) < 0.003
        x = torch.where(flip, 1.0 - x, x)
    return x


# ---- dataset -----------------------------------------------------------------------------
class CrohmeDataset(Dataset):
    """One split of the rendered CROHME data.

    item k -> (image tensor (1, 64, W), token indices <start> .. <end>, k)
    The CROHME renders have DARK ink on white paper, so ink_white=False.
    """

    def __init__(self, split, vocab, train=False, limit=None, data=DATA):
        self.data = Path(data)
        self.train = train
        self.items = []                     # (image path, true tokens, token indices)
        skipped = 0
        for image, latex in read_truth_tsv(split, data):
            tokens = tokenize(latex)
            if train and len(tokens) + 1 > MAX_LEN:    # too long to learn, leave out
                skipped += 1
                continue
            self.items.append((image, tokens, vocab.encode(tokens)))
        if limit:
            self.items = self.items[:limit]
        msg = f" ({skipped} longer than {MAX_LEN} tokens skipped)" if skipped else ""
        print(f"{split}: {len(self.items)} expressions{msg}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        image, _, ids = self.items[k]
        img = cv2.imread(str(self.data / image), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.data / image)
        if self.train:
            x = augment_tensor(prepare_image(img, ink_white=False, augment=augment_mask))
        else:
            x = prepare_image(img, ink_white=False)
        return x, torch.tensor(ids, dtype=torch.long), k


def collate(batch):
    """List of samples -> one batch.

    Images are padded on the right with paper up to the widest image; token
    lists are padded with <pad> up to the longest list (the loss ignores <pad>).
    """
    images, ids, keys = zip(*batch)
    x, widths = pad_images(images)
    tokens = torch.full((len(ids), max(len(t) for t in ids)), PAD, dtype=torch.long)
    for k, t in enumerate(ids):
        tokens[k, :len(t)] = t
    return x, widths, tokens, torch.tensor(keys)


# ---- metrics --------------------------------------------------------------------------------
def edit_distance(a, b):
    """Smallest number of token insertions, deletions and substitutions that turn a into b."""
    prev = list(range(len(b) + 1))            # distances for an empty prefix of a
    for i, ta in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,                 # delete ta
                         cur[j - 1] + 1,              # insert tb
                         prev[j - 1] + (ta != tb))    # keep (0) or substitute (1)
        prev = cur
    return prev[-1]


def score(predictions, truths):
    """Lists of token lists -> dict with ExpRate, <=1, <=2 (%), CER and token accuracy (%)."""
    dists = [edit_distance(p, t) for p, t in zip(predictions, truths)]
    n = max(len(dists), 1)
    cer = sum(dists) / max(sum(len(t) for t in truths), 1)
    return {
        "exprate": 100.0 * sum(d == 0 for d in dists) / n,
        "le1": 100.0 * sum(d <= 1 for d in dists) / n,
        "le2": 100.0 * sum(d <= 2 for d in dists) / n,
        "cer": cer,
        "token_acc": 100.0 * max(0.0, 1.0 - cer),
    }


@torch.no_grad()
def evaluate(model, loader, vocab, device):
    """Greedy-decode a whole split (the model uses its own guesses, no teacher forcing).

    Returns (metrics dict, rows) with rows = [(image, truth, prediction, edit distance)].
    """
    model.eval()                                       # BatchNorm/dropout in test mode
    dataset = loader.dataset
    preds = {}
    for x, widths, _, keys in loader:
        out = model.greedy_decode(x.to(device), widths.to(device), MAX_LEN)
        for k, ids in zip(keys.tolist(), out.cpu().tolist()):
            preds[k] = vocab.decode(ids)               # indices -> tokens, cut at <end>

    truths = [dataset.items[k][1] for k in range(len(dataset))]
    predictions = [preds[k] for k in range(len(dataset))]
    rows = [(dataset.items[k][0], " ".join(t), " ".join(p), edit_distance(p, t))
            for k, (t, p) in enumerate(zip(truths, predictions))]
    return score(predictions, truths), rows


# ---- loss: one training epoch, and the same loss on valid -------------------------------------------
def batch_loss(model, x, widths, tokens, vocab_size):
    """Cross-entropy of one batch with teacher forcing.

    Returns (mean loss per real token, number of real tokens).
    """
    # Teacher forcing: input = <start> y1 .. y_{L-1}, target = y1 .. y_L <end>
    logits = model(x, widths, tokens[:, :-1])          # (B, T, V)
    target = tokens[:, 1:]                             # (B, T)
    # Cross-entropy = -log(probability given to the right token), averaged
    # over all real tokens (<pad> positions are ignored).
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1),
                           ignore_index=PAD)
    return loss, int((target != PAD).sum())


def train_one_epoch(model, loader, optimizer, device, vocab_size, clip=5.0):
    """One pass over the training set. Returns the mean loss per token."""
    model.train()                                      # BatchNorm/dropout in training mode
    total_loss, total_tokens = 0.0, 0
    for x, widths, tokens, _ in loader:
        x, widths, tokens = x.to(device), widths.to(device), tokens.to(device)
        loss, n = batch_loss(model, x, widths, tokens, vocab_size)

        optimizer.zero_grad()
        loss.backward()                                # gradients for CNN, BiLSTMs, attention, decoder
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # avoid huge jumps
        optimizer.step()                               # update all weights

        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def eval_loss(model, loader, device, vocab_size):
    """Same loss as in training, but no weight updates and no augmentation
    (the loader's dataset has train=False). Returns the mean loss per token."""
    model.eval()                                       # BatchNorm/dropout in test mode
    total_loss, total_tokens = 0.0, 0
    for x, widths, tokens, _ in loader:
        x, widths, tokens = x.to(device), widths.to(device), tokens.to(device)
        loss, n = batch_loss(model, x, widths, tokens, vocab_size)
        total_loss += loss.item() * n
        total_tokens += n
    return total_loss / max(total_tokens, 1)


# ---- saving: checkpoints, history, plot -------------------------------------------------------------
def save_checkpoint(path, model, vocab, epoch, metrics):
    """Weights + everything needed to rebuild the model (config, vocab list)."""
    ckpt = {"model": model.state_dict(), "config": model.config, "vocab": vocab.itos,
            "epoch": epoch, "metrics": metrics}
    torch.save(ckpt, path)


HISTORY_FIELDS = ["epoch", "lr", "train_loss", "valid_loss",
                  "train_exprate", "train_le1", "train_le2", "train_cer", "train_token_acc",
                  "valid_exprate", "valid_le1", "valid_le2", "valid_cer", "valid_token_acc",
                  "seconds"]


def save_history(history, path):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def plot_history(history, path, best_epoch=None):
    """3 panels vs epoch, each train vs valid: loss | ExpRate | token accuracy.

    A growing gap (train much better than valid) means overfitting.
    """
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [("loss", "Loss (cross-entropy per token)", "loss"),
              ("exprate", "Expression accuracy (ExpRate, exact match)", "%"),
              ("token_acc", "Token accuracy (1 - CER)", "%")]
    for ax, (key, title, unit) in zip(axes, panels):
        ax.plot(ep, [h[f"train_{key}"] for h in history], "o-", color="tab:orange",
                label="train" if key == "loss" else "train (fixed subset)")
        ax.plot(ep, [h[f"valid_{key}"] for h in history], "o-", color="tab:blue", label="valid")
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if best_epoch is not None:                     # dashed line = epoch of best.pt
            ax.axvline(best_epoch, color="tab:green", ls="--", lw=1, label="best (valid CER)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(m):
    """Metrics dict -> short text."""
    return (f"ExpRate {m['exprate']:5.1f}% (<=1 {m['le1']:5.1f}%, <=2 {m['le2']:5.1f}%) | "
            f"CER {m['cer']:.3f} | token acc {m['token_acc']:5.1f}%")


# ---- main -------------------------------------------------------------------------------------------------
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train the CNN + BiLSTM + attention model.")
    ap.add_argument("--epochs", type=int, default=100, help="maximum number of epochs")
    ap.add_argument("--batch", type=int, default=32, help="images per training step")
    ap.add_argument("--eval-batch", type=int, default=32, help="images per step when decoding")
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    ap.add_argument("--patience", type=int, default=10,
                    help="stop after this many epochs without a better valid CER")
    ap.add_argument("--lr-patience", type=int, default=4,
                    help="divide the learning rate by 10 after this many epochs without progress")
    ap.add_argument("--coverage", action="store_true", help="use attention coverage")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout before the output layer")
    ap.add_argument("--enc-dropout", type=float, default=0.0,
                    help="dropout in the encoder: after the row BiLSTM and on the final grid G")
    ap.add_argument("--workers", type=int, default=2, help="processes that load images")
    ap.add_argument("--device", default=None, help="force cuda / mps / cpu")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N images of each split (quick test)")
    ap.add_argument("--train-eval-size", type=int, default=1000,
                    help="train images (fixed random subset) decoded every epoch for train metrics")
    ap.add_argument("--valid-eval-size", type=int, default=300,
                    help="valid images (fixed random subset) decoded every epoch; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT), help="folder for weights, history and plots")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Same random numbers every run -> results can be repeated.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Which device? (printed before anything else happens)
    device = get_device(args.device, verbose=True)

    # 2) Data
    vocab = Vocab.load(VOCAB_PATH)
    print(f"Vocabulary: {len(vocab)} tokens ({VOCAB_PATH.relative_to(ROOT)})")
    train_set = CrohmeDataset("train", vocab, train=True, limit=args.limit)
    valid_set = CrohmeDataset("valid", vocab, limit=args.limit)
    test_set = CrohmeDataset("test", vocab, limit=args.limit)   # used only after training

    # Fixed train subset for the per-epoch train metrics: picked ONCE (seeded),
    # the same images every epoch, and without augmentation (train=False), so
    # its numbers are directly comparable with valid.
    train_eval_set = copy.copy(train_set)
    train_eval_set.train = False
    n_sub = min(args.train_eval_size, len(train_set.items))
    train_eval_set.items = random.Random(args.seed).sample(train_set.items, n_sub)
    print(f"train subset for metrics: {n_sub} expressions (fixed, no augmentation)")

    # Fixed valid subset, decoded every epoch (greedy decoding is slow). It picks
    # the best weights and drives the learning rate and early stopping. The FULL
    # valid set is decoded once at the end. (valid loss still uses all of valid.)
    valid_eval_set = copy.copy(valid_set)
    n_val = len(valid_set.items)
    if args.valid_eval_size > 0:
        n_val = min(args.valid_eval_size, n_val)
    valid_eval_set.items = random.Random(args.seed).sample(valid_set.items, n_val)
    print(f"valid subset for metrics: {n_val} expressions (fixed)")

    loader_kw = dict(collate_fn=collate, num_workers=args.workers,
                     pin_memory=device.type == "cuda")
    # shuffle=True: a new random order (and new augmentation) every epoch
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, **loader_kw)
    train_eval_loader = DataLoader(train_eval_set, batch_size=args.eval_batch, shuffle=False,
                                   **loader_kw)
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch, shuffle=False, **loader_kw)
    valid_eval_loader = DataLoader(valid_eval_set, batch_size=args.eval_batch, shuffle=False,
                                   **loader_kw)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch, shuffle=False, **loader_kw)

    # 3) Model, optimizer, learning-rate schedule
    model = Im2LatexModel(len(vocab), use_coverage=args.coverage, dropout=args.dropout,
                          enc_dropout=args.enc_dropout).to(device)
    print(f"Model: {count_parameters(model) / 1e6:.2f} M parameters | "
          f"CNN channels {'-'.join(map(str, CNN_CHANNELS))} | coverage {args.coverage}")
    print(f"Dropout: encoder {args.enc_dropout} (row BiLSTM output + G) | "
          f"decoder {args.dropout} (before the output layer)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Divide the learning rate by 10 when the valid CER stops going down.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=args.lr_patience)

    # Everything is kept in memory while training; nothing is written to
    # outputs/ until the training is over.
    history = []                  # one dict of numbers per epoch
    best_cer, best_epoch, bad_epochs = float("inf"), None, 0
    best_weights = None           # copy of the weights of the best epoch so far
    best_metrics = None

    # 4) Training loop (Ctrl+C stops it early; the results so far are still saved)
    print(f"Training: up to {args.epochs} epochs, batch {args.batch}, lr {args.lr}")
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, optimizer, device, len(vocab))
            valid_loss = eval_loss(model, valid_loader, device, len(vocab))
            train_m, _ = evaluate(model, train_eval_loader, vocab, device)   # fixed subset
            valid_m, _ = evaluate(model, valid_eval_loader, vocab, device)   # fixed subset
            scheduler.step(valid_m["cer"])

            record = {"epoch": epoch, "lr": lr, "train_loss": train_loss,
                      "valid_loss": valid_loss,
                      **{f"train_{k}": v for k, v in train_m.items()},
                      **{f"valid_{k}": v for k, v in valid_m.items()},
                      "seconds": time.time() - t0}
            history.append(record)

            # Remember the weights of the epoch with the lowest VALID CER.
            # (a copy on the CPU, otherwise the next epoch would overwrite them)
            if valid_m["cer"] < best_cer:
                best_cer, best_epoch, bad_epochs = valid_m["cer"], epoch, 0
                best_weights = {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()}
                best_metrics = {"valid": valid_m}
                tag = "  <- best so far"
            else:
                bad_epochs += 1
                tag = f"  (no improvement for {bad_epochs})"

            print(f"epoch {epoch:3d} | lr {lr:.0e} | train loss {train_loss:.4f} | "
                  f"valid loss {valid_loss:.4f} | {record['seconds']:.0f}s{tag}")
            print(f"   train subset: {fmt(train_m)}")
            print(f"   valid subset: {fmt(valid_m)}")

            if bad_epochs >= args.patience:
                print(f"Early stopping: valid CER did not improve for {args.patience} epochs.")
                break
    except KeyboardInterrupt:
        print("\nTraining stopped by hand (Ctrl+C) - saving the results so far.")

    if best_weights is None:
        print("No epoch finished - nothing to save.")
        return

    # 5) Training is over: final result with the BEST weights (chosen on valid, never on test)
    model.load_state_dict(best_weights)
    train_m, _ = evaluate(model, train_eval_loader, vocab, device)
    valid_m, _ = evaluate(model, valid_loader, vocab, device)         # FULL valid set
    test_m, test_rows = evaluate(model, test_loader, vocab, device)   # the only test run
    print(f"\nBest epoch: {best_epoch}")
    print(f"   TRAIN SUBSET: {fmt(train_m)}")
    print(f"   VALID (all) : {fmt(valid_m)}")
    print(f"   TEST        : {fmt(test_m)}")

    # 6) Now write everything to outputs/model/
    save_checkpoint(out / "best.pt", model, vocab, best_epoch, best_metrics)
    save_history(history, out / "history.csv")
    plot_history(history, out / "accuracy_vs_epoch.png", best_epoch)
    with open(out / "test_results.json", "w") as fh:
        json.dump({"best_epoch": best_epoch, "model_config": model.config,
                   "train_subset": train_m, "valid": valid_m, "test": test_m}, fh, indent=2)
    with open(out / "test_predictions.tsv", "w", encoding="utf-8") as fh:
        fh.write("image\ttruth\tprediction\tedit_distance\n")
        for image, truth, pred, dist in test_rows:
            fh.write(f"{image}\t{truth}\t{pred}\t{dist}\n")
    print(f"Saved best.pt, history.csv, accuracy_vs_epoch.png, test_results.json, "
          f"test_predictions.tsv to {out}")

if __name__ == "__main__":
    main()
