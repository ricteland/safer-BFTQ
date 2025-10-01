import logging
import os
from torch.utils.tensorboard import SummaryWriter

def configure_logger(name, log_dir='logs'):
    """Configure a logger to save to a file and print to console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

class TensorBoardLogger:
    """A simple wrapper for torch.utils.tensorboard.SummaryWriter."""
    def __init__(self, log_dir='logs/tensorboard'):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()
