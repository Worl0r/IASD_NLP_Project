import glob
import json
import os
from datetime import datetime

import pandas as pd
import torch

import libconfig

from .types import Optimizer, Scheduler
from .utils import get_dict_from_pyfile, get_logger, setup_save_logs

logger = get_logger()


class DirectoryManager:
    def __init__(
        self,
        listOfExt,
        kindOfData,
        kindOfModel,
        modelName,
        dataName,
        device,
        id_config,
        start_rootine=False,
    ) -> None:
        self.kindOfData = kindOfData
        self.kindOfModel = kindOfModel
        self.listOfExt = listOfExt
        self.modelName = modelName
        self.dataName = dataName
        self.device = device
        self.id_config = id_config

        current_dir = os.getcwd()

        self.project_dir = current_dir

        self.dataExt = self.listOfExt[0]

        time_now = datetime.now()
        self.timestamp = time_now.strftime("%Y%m%d-%H%M%S")

        self.id_output_archive = None

        # Rootine
        if start_rootine:
            self.rootine()

    def rootine(self):
        logger.info("We start the routine.")

        self.check_organization_dir()
        logger.info("The organization of the directories is correct.")

        setup_save_logs(logger, self.get_logs_path())
        logger.info("The logs are saved in the logs directory.")

    # Archiving
    def archive_all(self, first_time):
        self.archive("output", first_time)

        self.archive("results", first_time)

    def load_model_from_archive(self, archive_output_path):
        self.unarchive("output", archive_output_path)

        archive_results_path = archive_output_path.replace("output", "results")
        archive_results_path = archive_results_path.replace(
            self.dataName + "/", ""
        )

        self.unarchive("results", archive_results_path)

    def archive(self, nature, first_time=False):
        current_model_path = self.choose_current_path(nature)

        model_path = os.path.dirname(current_model_path)

        # Check if there is the archive file
        archive_path = self.check_archive_directory(model_path)

        if os.path.exists(os.path.join(current_model_path, "archive_info")):
            if first_time:
                logger.warning(
                    "the archive_info already"
                    "exists in the current model path. first_time arg is wrong"
                )

            # copy the id_archive from archive_info in the current model path
            id_archive = self.return_archive_info(current_model_path)

        else:
            if nature == "output":
                id_archive = self.generate_id_archive(nature)
                self.id_output_archive = id_archive

            else:  # results mode
                if self.id_output_archive is None:
                    logger.error(
                        "The id_archive is not defined. You need to define it."
                    )
                    raise ValueError(
                        "The id_archive is not defined. You need to define it."
                    )

                id_archive = self.id_output_archive
                id_archive = id_archive.replace("output", "results")

        archived_model_path = os.path.join(
            archive_path,
            id_archive,
        )

        if first_time and nature == "output":
            # Save the id_archive in the current model path
            self.write_archive_output_info(current_model_path)

            # copy the configuration file in current model path
            if nature == "output":
                self.copy_configuration(current_model_path)

        elif first_time and nature == "results":
            # Save the id_archive in the current model path
            self.write_archive_input_info(current_model_path, id_archive)

        # Move the model
        os.rename(
            current_model_path,
            archived_model_path,
        )

        logger.info(
            f"The output directory called "
            f"{os.path.basename(archived_model_path)} "
            f"has been archived in {archived_model_path}."
        )

    def choose_current_path(self, nature):
        if nature == "output":
            current_path = self.get_checkpoint_dir_path()
        elif nature == "results":
            current_path = self.get_results_path()
        else:
            logger.error("The nature of the archiving is not correct.")
            raise ValueError("The nature of the archiving is not correct.")

        return current_path

    def write_archive_output_info(self, current_path):
        with open(os.path.join(current_path, "archive_info"), "w") as file:
            file.write(self.generate_id_archive("output"))

    def write_archive_input_info(self, current_path, id_archive):
        with open(os.path.join(current_path, "archive_info"), "w") as file:
            file.write(id_archive)

    def return_archive_info(self, current_path):
        with open(os.path.join(current_path, "archive_info")) as file:
            return file.read()

    def copy_configuration(self, current_model_path):
        config_path = os.path.join(self.project_dir, "configuration.py")
        config_archive_path = os.path.join(
            current_model_path, "configuration.py"
        )
        os.system(f"cp {config_path} {config_archive_path}")

    def unarchive(self, nature, archive_path):
        current_model_path = self.choose_current_path(nature)

        # if there is already an current_model_path directory we erase it
        if os.path.exists(current_model_path):
            os.system(f"rm -r {current_model_path}")
            logger.info(
                f"The output directory called "
                f"{os.path.basename(current_model_path)} "
                f"has been removed."
            )

        # Move the model
        os.rename(
            archive_path,
            current_model_path,
        )

        logger.info(
            f"The output directory called "
            f"{os.path.basename(archive_path)} "
            f"has been unarchived in {current_model_path}."
        )

    def get_archive_path(self):
        return os.path.join(self.project_dir, "archive")

    def check_archive_directory(self, dir_path):
        path = os.path.join(dir_path, "archive")

        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(
                f'The "archive" directory has been created in {dir_path}.'
            )

        return path

    def generate_id_archive(self, nature):
        list_keywords = [
            nature,
            self.dataName,
            self.modelName,
            self.id_config,
            self.timestamp,
        ]
        transformed_list = [str(word) + "_" for word in list_keywords[:-1]] + [
            str(list_keywords[-1])
        ]

        return "".join(transformed_list)

    # Savings
    def save_checkpoint(
        self,
        epoch: int,
        time_epoch: float,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: Scheduler,
        name: str,
        prefix: str = "",
        metrics: dict | None = None,
        module=None,
    ) -> None:
        """
        Checkpoint saver. Each save overwrites previous save.

        :param epoch: epoch number (0-indexed)
        :param model: transformer model
        :param optimizer: optimized
        :param prefix: checkpoint filename prefix
        """
        config = get_dict_from_pyfile(module)

        now = datetime.now()

        filename = self.get_checkpoint_path(prefix, name)

        if epoch != 0 and os.path.isfile(filename):
            checkpoint = torch.load(filename, weights_only=False)
            if checkpoint["epoch"] > epoch:
                logger.warning(f"Checkpoint already exists for epoch {epoch}.")

            sum_time = checkpoint["training_duration"] + time_epoch
        else:
            sum_time = time_epoch

        state = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "metrics": metrics,
            "configuration": config,
            "datetime": now.strftime("%m/%d/%Y, %H:%M:%S"),
            "training_duration": sum_time,
        }

        torch.save(state, filename)

    def save_data_predictions(self, groups, targets, predictions, time_idx):
        self.check_shapes(groups, targets)
        self.check_shapes(groups, predictions)
        self.check_shapes(groups, time_idx)

        # Round some ids
        # groups = groups.round()
        # time_idx = time_idx.round()
        predictions = predictions.round()

        merge_targets_predictions = torch.cat((targets, predictions), 1)
        group_data = torch.cat((groups, merge_targets_predictions), 1)
        time_data = torch.cat((time_idx, group_data), 1)

        path = os.path.join(
            self.get_results_path(), self.dataName + "_data_predictions.csv"
        )

        pd.DataFrame(time_data.cpu()).to_csv(
            path, header=self.get_columns_name(), index=True
        )

        logger.info(f"Data predictions saved in {path}.")

    def save_evaluation_metrics(self, metrics):
        path = os.path.join(
            self.get_results_path(), self.dataName + "_evaluation_metrics.txt"
        )

        with open(path, "w") as file:
            file.write(json.dumps(metrics))

        logger.info(f"Metrics saved in {path}.")

    ## Figures
    def save_figures(self, figs, name):
        for i, fig in enumerate(figs):
            name_idx = name + "_" + str(i) + ".png"
            self.save_figure(fig, name_idx)

    def save_figure(self, fig, name):
        path = os.path.join(self.get_results_figure_path(), name)
        fig.savefig(path)
        logger.info(f"Figure saved in {path}.")

    # Logs
    def get_logs_path(self):
        path = os.path.join(
            self.project_dir,
            "logs",
            self.dataName,
            self.modelName,
            "logs_" + self.timestamp + ".txt",
        )

        return path

    def clear_saved_logs(
        self,
        keepLlast: bool = False,
        keepThisOne: str | None = None,
    ):
        logs_path = self.get_logs_path()
        dir_path = os.path.dirname(logs_path)
        list_files = glob.glob(os.path.join(dir_path, "*.txt"))

        if keepLlast:
            list_files = list_files[:-1]

        if keepThisOne is not None:
            list_files = [
                file for file in list_files if keepThisOne not in file
            ]

        for file in list_files:
            os.remove(file)
            logger.info(
                f"The file {os.path.basename(file)} has been removed from {dir_path}."
            )

    # Checkings
    def check_organization_dir(
        self,
    ) -> None:
        # Check of the ouput's organization
        self.check_dir_tree("output")

        # Check of the input's organization
        self.check_dir_tree("input", skipModelDir=True)

        # Check of the results's organization
        self.check_sub_dir(
            path=self.project_dir,
            name="results",
            kind=["results"],
            check=self.check_dir,
            callback=self.check_sub_dir,
            kwargs={
                "name": self.dataName,
                "kind": [self.dataName],
                "callback": self.check_sub_dir,
                "kwargs": {
                    "name": "figures",
                    "kind": ["figures"],
                },
            },
        )

        # Check of the log's organization
        self.check_dir_tree("logs")

        # Check a specific input file
        self.check_dir_tree(
            "input", skipModelDir=True, check=self.check_file_data
        )

    def check_dir_tree(self, treeName, skipModelDir=False, check=None):
        if check is None:
            check = self.check_dir

        dir = os.path.join(self.project_dir, treeName)

        self.check_dir(dir)

        # Check the type of data
        if skipModelDir:
            self.check_sub_dir(dir, self.dataName, self.kindOfData, check)
        else:
            self.check_sub_dir(
                dir,
                self.dataName,
                self.kindOfData,
                check,
                self.check_sub_dir,
                (self.modelName, self.kindOfModel),
            )

    def check_sub_dir(
        self, path, name, kind, check, callback=None, *args, **kwargs
    ):
        if name in kind:
            dir = os.path.join(path, name)
            check(dir)
            if callback is not None:
                if (
                    (len(args) != 0 and len(args[0]))
                    or ("kwargs" in kwargs and len(kwargs["kwargs"]))
                ) < 2:
                    logger.error(
                        "The callback function needs at least two arguments."
                    )
                    raise ValueError(
                        "The callback function needs at least two arguments."
                    )
                if len(args) != 0 and "kwargs" not in kwargs:
                    callback(dir, *args[0], check=check)
                elif len(args) != 0 and "kwargs" in kwargs:
                    callback(dir, *args[0], **kwargs["kwargs"], check=check)
                elif len(args) == 0 and "kwargs" in kwargs:
                    callback(
                        dir,
                        **kwargs["kwargs"],
                        check=check,
                    )
                else:
                    logger.error(
                        "Something went wrong with the callback function."
                    )
                    raise ValueError(
                        "Something went wrong with the callback function."
                    )

    def check_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            name = os.path.basename(path)
            path = os.path.dirname(path)
            logger.info(f'The "{name}" directory has been created in {path}.')

    def check_file_data(self, path):
        seqBool = [
            os.path.exists(self.get_input_path(ext)) for ext in self.listOfExt
        ]

        if not any(seqBool):
            name = os.path.basename(path)
            path = os.path.dirname(path)
            logger.error(f'The "{name}" file does not exist in {path}.')
            raise FileNotFoundError(
                f'The "{name}" file does not exist in {path}.'
            )
        else:
            self.dataExt = self.listOfExt[seqBool.index(True)]

    # Get methods
    def get_checkpoint_dir_path(self):
        return os.path.join("./output", self.dataName, self.modelName)

    def get_checkpoint_path(self, name, prefix=""):
        path = self.get_checkpoint_dir_path()
        return os.path.join(path, prefix + name + ".pth.tar")

    def get_input_path(self, ext=None):
        if ext is None:
            ext = self.dataExt
        return os.path.join(
            "./input", self.dataName, self.dataName + "." + ext
        )

    def get_utils_input_path(self):
        return os.path.join(
            os.path.dirname(self.get_input_path()),
            libconfig.inputStructure[self.dataName]["utils"],
        )

    def get_results_path(self):
        return os.path.join(
            self.project_dir,
            "results",
            self.dataName,
        )

    def get_results_figure_path(self):
        return os.path.join(
            self.get_results_path(),
            "figures",
        )

    ## Get libconfig
    def get_columns_name(self) -> list[str]:
        return libconfig.outputStructure[self.dataName]["columns"]

    ## Get checkpoint
    def get_config_checkopint(self, name):
        path = self.get_checkpoint_path(name)
        checkpoint = torch.load(
            path, map_location=torch.device(self.device), weights_only=False
        )
        return checkpoint["configuration"]

    # Utils
    def check_shapes(self, first, second):
        if first.shape[0] != second.shape[0]:
            logger.error("The tensor shapes are different.")
            raise ValueError("The tensor shapes are different.")
