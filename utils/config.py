import torch



MAX_SEQ_LEN = 50
IMG_SIZE = 224
TEXT_MODEL = "bert-base-uncased"
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
