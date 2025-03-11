import logging
import os
import yaml


def read_yaml_config(path):
    try:
        with open(path) as file:
            content = yaml.safe_load(file)
            return content
    except FileNotFoundError:
        print(f"The file {path} does not exist.")
    except yaml.YAMLError as e:
        print(f"Error on the reading of the file : {e}")


def get_number_parameters(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"The model has {trainable_params} trainable parameters")


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
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
        logger.info("We created the logs file in {logs_path}.")

    file_handler = logging.FileHandler(logs_path)
    # Format des logs
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
