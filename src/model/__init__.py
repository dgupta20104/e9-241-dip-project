"""Recognition model (CNN + row/column BiLSTM + 2-D attention decoder): line image -> LaTeX.

    vocab.py    LaTeX tokenizer + vocabulary (token <-> index), builds data/processed/crohme/vocab.txt
    model.py    the network (Im2LatexModel) + the shared image preprocessing (prepare_image)
    train.py    training script (saves weights, history and accuracy plots to outputs/model/)
    predict.py  load trained weights and turn line crops / images into LaTeX
"""
