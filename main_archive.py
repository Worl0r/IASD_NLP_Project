import os
from sys import argv

import configuration
import libconfig
from play_with_transformers import DirectoryManager, get_logger

logger = get_logger()

ID_CONFIG = configuration.ID_CONFIGURATION


def archive_model():
    DirMana = DirectoryManager(
        libconfig.listOfExt,
        libconfig.kindOfData,
        libconfig.kindOfModel,
        configuration.model_name,
        configuration.data_name,
        configuration.device,
        configuration.ID_CONFIGURATION,
    )

    logger.info("Starting the program : main_archive.")
    DirMana.rootine()

    # Archive the model
    logger.info("We are going to plot the training and evaluation results.")
    DirMana.archive_all(first_time=False)


def load_archive(TIME_STAMP):
    DirMana = DirectoryManager(
        libconfig.listOfExt,
        libconfig.kindOfData,
        libconfig.kindOfModel,
        configuration.model_name,
        configuration.data_name,
        configuration.device,
        configuration.ID_CONFIGURATION,
    )

    logger.info("Starting the program : main_archive.")
    DirMana.rootine()

    # Name
    archive_name = (
        "output_"
        + configuration.data_name
        + "_"
        + configuration.model_name
        + "_"
        + str(ID_CONFIG)
        + "_"
        + TIME_STAMP
    )

    # Ouput Path
    current_model_path = DirMana.choose_current_path("output")
    model_path = os.path.dirname(current_model_path)
    archive_output_path = DirMana.check_archive_directory(model_path)
    archive_output_path = os.path.join(archive_output_path, archive_name)

    DirMana.load_model_from_archive(archive_output_path)

    logger.info("We load the archive called:" + archive_output_path)


def delete_archive(TIME_STAMP):
    DirMana = DirectoryManager(
        libconfig.listOfExt,
        libconfig.kindOfData,
        libconfig.kindOfModel,
        configuration.model_name,
        configuration.data_name,
        configuration.device,
        configuration.ID_CONFIGURATION,
    )

    logger.info("Starting the program : main_archive.")
    DirMana.rootine()

    # Name
    archive_name = (
        "output_"
        + configuration.data_name
        + "_"
        + configuration.model_name
        + "_"
        + str(ID_CONFIG)
        + "_"
        + TIME_STAMP
    )

    # Ouput Path
    current_model_path = DirMana.choose_current_path("output")

    model_path = os.path.dirname(current_model_path)

    archive_output_path = DirMana.check_archive_directory(model_path)
    archive_output_path = os.path.join(archive_output_path, archive_name)

    logger.info(f"Archive output path: {archive_output_path}")
    logger.info(f"Model path: {model_path}")

    logger.info("We delete the archive in the output path.")
    # remove all the directory
    os.system(f"rm -rf {archive_output_path}")

    # Input Path
    current_model_path = DirMana.choose_current_path("results")

    model_path = os.path.dirname(current_model_path)

    archive_results_path = DirMana.check_archive_directory(model_path)
    archive_results_path = os.path.join(
        archive_results_path, archive_name.replace("output", "results")
    )

    logger.info(f"Archive results path: {archive_results_path}")

    logger.info("We delete the archive in the restults path.")
    # remove all the directory
    os.system(f"rm -rf {archive_results_path}")


def main(mode, TIME_STAMP):
    if mode == "archive":
        archive_model()
    elif mode == "load":
        load_archive(TIME_STAMP)
    elif mode == "delete":
        delete_archive(TIME_STAMP)
    else:
        logger.error("Mode not recognized.")
        raise ValueError("Mode not recognized.")


if __name__ == "__main__":
    logger.info("Starting the program : main_archive.")

    if len(argv) == 1:
        logger.error("No argument given.")
        raise ValueError("No argument given")

    elif len(argv) <= 5:
        if argv[1] != "-m":
            logger.error(f'Unrecognized argument "{argv[1]}"')
            raise ValueError(f'Unrecognized argument "{argv[1]}"')

        mode = argv[2]

        TIME_STAMP = argv[4] if (len(argv) >= 4 and argv[3] == "-t") else None

        main(mode, TIME_STAMP)
    else:
        logger.error("Wrong number of arguments")

    logger.info("End of the program : main_archive.")
