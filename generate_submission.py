import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import torch
from more_itertools import bucket

from idao.data_module import IDAODataModule
from idao.model import SimpleConv


def compute_predictions(mode, dataloader, checkpoint_path, cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info("Loading checkpoint")
    model = SimpleConv.load_from_checkpoint(checkpoint_path, mode=mode)
    model = model.cpu().eval()

    dict_pred = defaultdict(list)
    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")

    # TODO(kazevn) batch predictions
    for img, name in iter(dataloader):
        if mode == "classification":
            dict_pred["id"].append(name[0].split('.')[0])
            output = (1 if torch.round(model(img)["class"].detach()[0][0]) == 0 else 0)
            dict_pred["particle"].append(output)

        else:
            output = model(img)["energy"].detach()
            dict_pred["energy"].append(output[0][0].item())
    return dict_pred


def main():
    config = configparser.ConfigParser()
    config.read("./config.ini")
    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=64, cfg=config
    )

    dataset_dm.prepare_data()
    dataset_dm.setup()
    dl = dataset_dm.test_dataloader()

    dict_pred = defaultdict(list)
    for mode in ["regression", "classification"]:
        if mode == "classification":
            model_path = config["REPORT"]["ClassificationCheckpoint"]
        else:
            model_path = config["REPORT"]["RegressionCheckpoint"]

        dict_pred.update(compute_predictions(mode, dl, model_path, cfg=config))

    data_frame = pd.DataFrame(dict_pred,
                              columns=["id", "energy", "particle"])
    data_frame.set_index("id", inplace=True)
    data_frame.to_csv('submission.csv.gz', index=True, header=True, index_label="id")


if __name__ == "__main__":
    main()
