import lightning as L

from .null_classifier import NullClassifier


class CobaldParser(L.LightningModule):
    def __init__(self, embedder, tagger):
        super().__init__()
        self.null_classifier = NullClassifier()
        self.tagger = tagger

