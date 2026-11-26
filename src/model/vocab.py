"""LaTeX tokenizer + vocabulary for the recognition model.

The decoder LSTM does not work with text, it works with numbers: every LaTeX
token (like `x`, `+`, `\\frac`, `{`) gets an index, and the model predicts one
index per step. This file

    1. splits a LaTeX string into tokens and cleans it up (`tokenize`),
    2. builds the token list from the TRAIN labels and saves it as
       data/processed/crohme/vocab.txt (`build_vocab`),
    3. converts tokens <-> indices (`Vocab`).

vocab.txt: one token per line, line number = index.
    0 <pad>    filler used to make all label sequences in a batch equally long
    1 <start>  first input of the decoder ("start writing now")
    2 <end>    the decoder outputs this when the expression is finished
    3 <unk>    any token that never appeared in the training labels
    4 ...      real tokens, most frequent first

Clean-up (why): the CROHME labels often contain markup that is invisible in
handwriting, e.g. `\\left( x \\right)` and `( x )` look exactly the same on paper.
The model cannot see this difference, so both are turned into the same tokens.
    * drop invisible markup: \\left \\right \\big \\Big \\bigg \\Bigg \\limits \\displaystyle,
      spacing \\! \\, \\; \\: \\(space) \\quad, a lone backslash, and `$`
    * unwrap font boxes: \\mbox{FfN} -> F f N (also \\mathrm, \\text)
    * merge look-alikes: < -> \\lt, > -> \\gt, \\lbrack -> [, ' -> \\prime, \\to -> \\rightarrow, ...
    * braces: arguments of ^ _ \\sqrt \\frac always get braces (x^2 -> x ^ { 2 }),
      other braces are removed ({x^{2}} -> x ^ { 2 }), sub-script before super-script
      (x^{2}_{i} -> x _ { i } ^ { 2 }), a bare prime becomes a super-script
      (f' -> f ^ { \\prime }).

Usage (from the project root):
    python -m src.model.vocab          # builds vocab.txt and prints a report

    from src.model.vocab import Vocab, tokenize
    vocab = Vocab.load()
    ids = vocab.encode(tokenize(r"\\frac{1}{x^2}"))
"""

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # project root (src/model/vocab.py -> ../..)
DATA = ROOT / "data" / "processed" / "crohme"       # rendered images + <split>_truth.tsv
VOCAB_PATH = DATA / "vocab.txt"

# ---- special tokens (their index = their position in this list) -------------
PAD, START, END, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<start>", "<end>", "<unk>"]

# ---- tokenizer rules ------------------------------------------------------------
# One token = a command (\frac, \alpha), an escaped character (\{, \!, "\ "),
# or any single non-space character (x, 2, +, {).
TOKEN_RE = re.compile(r"\\[A-Za-z]+|\\.|\S")

# Tokens that leave no visible mark on the paper -> thrown away.
DROP = {
    "$", "\\", "\\\\",                                   # math-mode signs, line breaks
    "\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg",   # bracket sizing
    "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl", "\\biggr", "\\Biggl", "\\Biggr",
    "\\limits", "\\nolimits", "\\displaystyle",          # layout hints
    "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad",     # spacing
}
SIZERS = {"\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg",
          "\\bigl", "\\bigr", "\\Bigl", "\\Bigr", "\\biggl", "\\biggr", "\\Biggl", "\\Biggr"}

# Different spellings of the same drawn symbol -> one spelling.
# A list value means "replace by several tokens" (\ltN is a typo for \lt N).
SAME_SYMBOL = {
    "<": "\\lt", ">": "\\gt",
    "\\lbrack": "[", "\\rbrack": "]",
    "\\lbrace": "\\{", "\\rbrace": "\\}",
    "'": "\\prime",
    "\\to": "\\rightarrow",
    "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq",
    "\\dots": "\\ldots",
    "\\ltN": ["\\lt", "N"],
}

# Font boxes: the box itself is invisible, its content is kept.
UNWRAP = {"\\mbox", "\\mathrm", "\\text", "\\textrm", "\\mathit", "\\mathbf"}

SCRIPTS = {"^", "_"}           # super-script, sub-script
PRIME = "\\prime"


def is_script(node):
    """True for a ^ or _ token (a Group is a list, so it can never be one)."""
    return isinstance(node, str) and node in SCRIPTS


def split_tokens(latex):
    """Step 1: raw string -> list of cleaned tokens (no brace handling yet)."""
    raw = TOKEN_RE.findall(latex)
    out = []
    i = 0
    while i < len(raw):
        tok = raw[i]
        # "\left." / "\right." is an invisible bracket: drop the dot as well
        if tok in SIZERS and i + 1 < len(raw) and raw[i + 1] == ".":
            i += 2
            continue
        if tok in DROP:
            i += 1
            continue
        new = SAME_SYMBOL.get(tok, tok)
        out.extend(new if isinstance(new, list) else [new])
        i += 1
    return out


# ---- braces: tiny tree parser ---------------------------------------------------
class Group(list):
    """The tokens between a `{` and its matching `}` (may contain more Groups)."""


def parse_groups(tokens):
    """Step 2: flat tokens -> nested list, every {...} becomes a Group.

    A `}` without an opening `{` is dropped; a `{` that is never closed is
    closed at the end. Font boxes (\\mbox, ...) are removed here, their
    content stays as a normal group.
    """
    root = Group()
    stack = [root]                       # stack[-1] = group we are currently filling
    for tok in tokens:
        if tok == "{":
            g = Group()
            stack[-1].append(g)
            stack.append(g)
        elif tok == "}":
            if len(stack) > 1:           # stray "}" at the top level -> ignore
                stack.pop()
        elif tok in UNWRAP:
            continue                     # \mbox{abc} -> {abc}
        else:
            stack[-1].append(tok)
    return root


def _take_atom(nodes, i):
    """Read one 'thing' starting at nodes[i] (without the scripts after it).

    Returns (tokens, next index, is_plain_group). A plain group comes back
    WITHOUT braces; the caller decides whether braces are needed.
    \\frac takes 2 arguments, \\sqrt takes an optional [index] and 1 argument;
    their arguments always get braces.
    """
    node = nodes[i]
    i += 1
    if isinstance(node, Group):
        return emit(node), i, True
    if node == "\\frac":
        out = ["\\frac"]
        for _ in range(2):                                   # numerator, denominator
            arg = []
            if i < len(nodes) and not is_script(nodes[i]):
                arg, i, _ = _take_atom(nodes, i)
            out += ["{", *arg, "}"]
        return out, i, False
    if node == "\\sqrt":
        out = ["\\sqrt"]
        if i < len(nodes) and nodes[i] == "[":               # \sqrt[3]{x}: root index
            j = i + 1
            while j < len(nodes) and nodes[j] != "]":
                j += 1
            if j < len(nodes):                               # found the closing "]"
                out += ["[", *emit(Group(nodes[i + 1:j])), "]"]
                i = j + 1
        arg = []
        if i < len(nodes) and not is_script(nodes[i]):
            arg, i, _ = _take_atom(nodes, i)
        out += ["{", *arg, "}"]
        return out, i, False
    return [node], i, False


def emit(nodes):
    """Step 3: nested groups -> flat token list in one fixed (canonical) form."""
    out = []
    i = 0
    while i < len(nodes):
        # A prime with nothing in front of it (the content of f^{'}) stays a
        # plain token; primes that follow a base are handled in step 2.
        if nodes[i] == PRIME:
            out.append(PRIME)
            i += 1
            continue

        # 1) the base, e.g. "x" in x^{2}; nothing if the line starts with ^ or _
        base, is_group = [], False
        if not is_script(nodes[i]):
            base, i, is_group = _take_atom(nodes, i)

        # 2) all scripts that follow the base: collect sub- and super-script content
        sub, sup = [], []
        while i < len(nodes) and (is_script(nodes[i]) or nodes[i] == PRIME):
            if nodes[i] == PRIME:                    # f' -> f ^ { \prime }
                sup.append(PRIME)
                i += 1
                continue
            kind = nodes[i]
            i += 1
            if i < len(nodes) and not is_script(nodes[i]):
                arg, i, _ = _take_atom(nodes, i)
                (sup if kind == "^" else sub).extend(arg)

        # 3) a plain {group} only keeps its braces if it is the base of a
        #    script and has more than one token: {x^{2}}^{3} vs {e}_{1} -> e_{1}
        if is_group and (sub or sup) and len(base) > 1:
            base = ["{", *base, "}"]
        out += base
        if sub:
            out += ["_", "{", *sub, "}"]
        if sup:
            out += ["^", "{", *sup, "}"]
    return out


def tokenize(latex):
    """LaTeX string -> list of clean tokens, e.g. 'x^2+\\left(1\\right)' ->
    ['x', '^', '{', '2', '}', '+', '(', '1', ')']."""
    return emit(parse_groups(split_tokens(latex)))


def to_latex(tokens):
    """Tokens -> LaTeX string (tokens joined by single spaces)."""
    return " ".join(tokens)


# ---- reading the label files ------------------------------------------------------
def read_truth_tsv(split, data=DATA):
    """data/processed/crohme/<split>_truth.tsv -> list of (image path, raw latex).

    The image path is relative to `data` (e.g. 'train/images/xyz.png').
    The first line of the file is a header and is skipped.
    """
    rows = []
    with open(Path(data) / f"{split}_truth.tsv", encoding="utf-8") as fh:
        next(fh)                                          # skip "image<TAB>latex"
        for line in fh:
            line = line.rstrip("\n")
            if line:
                image, _, latex = line.partition("\t")
                rows.append((image, latex))
    return rows


# ---- vocabulary: token <-> index ------------------------------------------------------
class Vocab:
    """Look-up tables between tokens and indices.

    vocab.itos[i] = token with index i   ("index to string")
    vocab.stoi[t] = index of token t     ("string to index")
    """

    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        assert self.itos[:4] == SPECIALS, "the first 4 tokens must be the special tokens"

    def __len__(self):
        return len(self.itos)

    @classmethod
    def load(cls, path=VOCAB_PATH):
        """Read vocab.txt (one token per line)."""
        with open(path, encoding="utf-8") as fh:
            return cls([line.rstrip("\n") for line in fh if line.rstrip("\n")])

    def save(self, path=VOCAB_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(self.itos) + "\n")

    def encode(self, tokens, add_start_end=True):
        """Tokens -> indices; unknown tokens -> <unk>; optionally wrapped in <start> ... <end>."""
        ids = [self.stoi.get(t, UNK) for t in tokens]
        return [START] + ids + [END] if add_start_end else ids

    def decode(self, ids):
        """Indices -> tokens. Stops at <end>, skips <pad> and <start>."""
        tokens = []
        for i in ids:
            i = int(i)
            if i == END:
                break
            if i in (PAD, START):
                continue
            tokens.append(self.itos[i] if 0 <= i < len(self.itos) else SPECIALS[UNK])
        return tokens


def build_vocab(data=DATA, path=VOCAB_PATH):
    """Count the tokens of all TRAIN labels and save vocab.txt.

    Order: the 4 special tokens, then tokens from most to least frequent
    (ties alphabetically, so the file is the same every time).
    Returns (Vocab, Counter of train tokens).
    """
    counts = Counter()
    for _, latex in read_truth_tsv("train", data):
        counts.update(tokenize(latex))
    tokens = sorted(counts, key=lambda t: (-counts[t], t))
    vocab = Vocab(SPECIALS + tokens)
    vocab.save(path)
    return vocab, counts


def main():
    vocab, counts = build_vocab()
    print(f"Saved {VOCAB_PATH.relative_to(ROOT)}: {len(vocab)} entries "
          f"(4 special + {len(vocab) - 4} tokens from train)")
    print("Tokens (train count):")
    print("  " + "  ".join(f"{t} {counts[t]}" for t in vocab.itos[4:]))

    # How well does the train vocabulary cover the other splits?
    for split in ("train", "valid", "test"):
        rows = read_truth_tsv(split)
        lengths, unknown, n_bad = [], Counter(), 0
        for _, latex in rows:
            toks = tokenize(latex)
            lengths.append(len(toks))
            missing = [t for t in toks if t not in vocab.stoi]
            unknown.update(missing)
            n_bad += bool(missing)
        lengths.sort()
        print(f"{split}: {len(rows)} expressions, tokens per expression: "
              f"mean {sum(lengths) / len(lengths):.1f}, max {lengths[-1]}, "
              f"> 150: {sum(n > 150 for n in lengths)} | "
              f"unknown tokens: {dict(unknown) or 'none'} in {n_bad} expressions")

    # A few labels that the clean-up changes, to check the rules by eye.
    print("Examples (raw -> tokens):")
    shown = 0
    for _, latex in read_truth_tsv("train"):
        if any(k in latex for k in ("\\left", "\\mbox", "\\limits", "^\\prime", "'", "{x^{2}}")):
            print(f"  {latex}\n    -> {to_latex(tokenize(latex))}")
            shown += 1
            if shown == 8:
                break


if __name__ == "__main__":
    sys.exit(main())
