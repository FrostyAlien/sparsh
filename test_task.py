# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf, DictConfig

from tactile_ssl.utils import configure_torch_backends, get_best_available_device


def resolve_experiment_dir(base_dir: str, cfg: DictConfig) -> str:
    config_path = os.path.join(base_dir, "config.yaml")
    if os.path.isfile(config_path):
        return base_dir

    exp_name = f"{cfg.sensor}_{cfg.task_name}_{cfg.ssl_name}_vit{cfg.ssl_model_size}_{cfg.train_data_budget}"
    direct_matches = []
    scored_matches = []
    name_tokens = [cfg.sensor, cfg.task_name, cfg.ssl_name, f"vit{cfg.ssl_model_size}"]

    for exp in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, exp)
        if not os.path.isdir(exp_dir):
            continue
        exp_config = os.path.join(exp_dir, "config.yaml")
        if not os.path.isfile(exp_config):
            continue
        if exp_name in exp and not exp.startswith("2024"):
            direct_matches.append(exp_dir)
        score = sum(token in exp for token in name_tokens)
        scored_matches.append((score, exp_dir))

    if direct_matches:
        return direct_matches[0]

    if scored_matches:
        scored_matches.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_dir = scored_matches[0]
        if best_score > 0:
            return best_dir

    raise FileNotFoundError(
        f"Could not find config.yaml under {base_dir}. "
        "Expected either a direct config file or a downloaded experiment subdirectory."
    )


def get_dataset_reskin(cfg: DictConfig):
    raise NotImplementedError("Not implemented yet.")


def get_dataset_digit(cfg: DictConfig, dataset_name: str):
    data_cfg = cfg.data
    test_cfg = cfg.test.data
    look_in_folder = test_cfg.get("look_in_folder", False)
    path_dataset = data_cfg.dataset.config.path_dataset

    if not look_in_folder:
        test_dset = hydra.utils.instantiate(data_cfg.dataset, dataset_name=dataset_name)
    else:
        datasets_list = os.listdir(os.path.join(path_dataset, dataset_name))
        test_dset = []
        for f in datasets_list:
            subdataset_name = dataset_name + "/" + f.split(".")[0]
            test_dset.append(hydra.utils.instantiate(data_cfg.dataset, dataset_name=subdataset_name))
        test_dset = data.ConcatDataset(test_dset)

    test_loader = data.DataLoader(
            test_dset,
            batch_size=cfg.test.data.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

    return test_dset, test_loader


def get_test_dataset(cfg: DictConfig, dataset_name: str):
    data_cfg = cfg.data

    if data_cfg.sensor == "digit" or "gelsight" in data_cfg.sensor:
        test_dset, test_loader = get_dataset_digit(cfg, dataset_name)
    elif data_cfg.sensor == "reskin":
        test_dset, test_loader = get_dataset_reskin(cfg)
    else:
        raise NotImplementedError("Sensor type not implemented yet.")
    return test_dset, test_loader


def test(cfg: DictConfig):
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)

    device = get_best_available_device()
    configure_torch_backends(device)

    print(f"Instantiating model <{cfg.task._target_}>")
    task_name = cfg.experiment_name
    path_checkpoints = cfg.paths.output_dir + "/checkpoints/"
    eval_ckpts = sorted(os.listdir(path_checkpoints))
    eval_ckpts = [ckpt for ckpt in eval_ckpts if ckpt[-4:] == ".pth"]

    if 'forcefield' in task_name:
        eval_ckpts = [eval_ckpts[-1]]

    for ckpt in eval_ckpts:
        cfg.task.checkpoint_task = f"{path_checkpoints}/{ckpt}"
        model = hydra.utils.instantiate(cfg.task)
        print(f"Testing {task_name}  - {ckpt}")
        tester_partial = hydra.utils.instantiate(cfg.test.tester)
        tester = tester_partial(device=device, module=model)

        for dataset_name in cfg.test.data.dataset_name:
            print(f"\t Testing on {dataset_name}")
            test_dset, test_dataloader = get_test_dataset(cfg, dataset_name)

            tester.set_test_params(
                task=task_name,
                sensor=cfg.sensor,
                ckpt=ckpt,
                dataset_name=dataset_name,
                path_outputs=cfg.test.path_outputs,
            )

            tester.run_model(test_dset, test_dataloader)
            tester.make_plots(test_dset)

        tester.get_overall_metrics(test_dset, over_all_outputs=True)


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    path_outputs = cfg.paths.output_dir

    path_outputs = resolve_experiment_dir(path_outputs, cfg)
    exp_config = f"{path_outputs}/config.yaml"

    test_cfg = cfg.test.copy()
    data = cfg.data.copy()
    cfg = OmegaConf.load(exp_config)
    cfg.data = data
    cfg.test = test_cfg
    cfg.paths.output_dir = path_outputs

    test(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
