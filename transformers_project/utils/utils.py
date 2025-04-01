import logging

import psutil
import yaml


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_dict_from_pyfile(module):
    if module is None:
        return {}

    listExcep = ["torch", "cudnn", "nn"]

    # Obtain all the variables from the module
    return {
        k: v
        for k, v in vars(module).items()
        if not callable(v) and not k.startswith("__")
        if k not in listExcep
    }


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, keepTrack: bool = False) -> None:
        self.keepTrack = keepTrack
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val: float | int, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.keepTrack:
            self.list.extend([val])


def get_logger() -> logging.Logger:
    # Create the config for the logger
    logging.basicConfig(
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Set the logger level of priority
    logging.getLogger().setLevel(logging.DEBUG)

    # Pick the name of the current file
    logger = logging.getLogger(__name__)

    return logger


def setup_save_logs(logger, logs_path):
    file_handler = logging.FileHandler(logs_path)
    # Format des logs
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_config(file_path):
    with open(file_path) as file:
        return yaml.safe_load(file)


def print_memory_usage():
    """Affiche l'utilisation de la mémoire du processus et du système."""
    logger = get_logger()

    # Mémoire du processus courant
    process_mem = psutil.Process().memory_info()
    logger.info(
        f"Utilisation mémoire du processus : {process_mem.rss / 1024 ** 2:.2f} MB"
    )

    # Mémoire système
    system_mem = psutil.virtual_memory()
    logger.info(
        f"Mémoire système - Totale: {system_mem.total / 1024 ** 3:.2f} Go, "
        f"Utilisée: {system_mem.used / 1024 ** 3:.2f} Go, "
        f"Disponible: {system_mem.available / 1024 ** 3:.2f} Go"
    )
