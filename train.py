import configparser
import pathlib as path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from idao.data_module import IDAODataModule
from idao.model import SimpleConv
from idao.model import SanteriConv


def get_free_gpu():
    """
    Returns the index of the GPU with the most free memory.
    Different from lightning's auto_select_gpus, as it provides the most free GPU, not an absolutely free.
    """
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
    nvmlInit()

    return np.argmax([
        nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(i)).free
        for i in range(nvmlDeviceGetCount())
    ])


def trainer(mode: ["classification", "regression"], cfg, dataset_dm):
    model = SanteriConv(mode=mode)
    print(model)
    input("Press Enter to continue...")
    if cfg.getboolean("TRAINING", "UseGPU"):
        gpus = [get_free_gpu()]
    else:
        gpus = None
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=int(epochs),
        progress_bar_refresh_rate=20,
        weights_save_path=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]).joinpath(
            mode
        ),
        default_root_dir=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]),
        auto_lr_find=True
    )

    # Tune the model
    trainer.tune(model,  dataset_dm)
    
    # Train the model ⚡
    trainer.fit(model, dataset_dm)


def main():
    seed_everything(666)
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()

#    for mode in ["classification", "regression"]:
    for mode in ["regression"]:
        print(f"Training for {mode}")
        trainer(mode, cfg=config, dataset_dm=dataset_dm)


if __name__ == "__main__":
    main()
