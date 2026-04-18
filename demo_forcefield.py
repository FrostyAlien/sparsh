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


def demo(cfg: DictConfig):
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
    last_ckpt = eval_ckpts[-1] #-3

    cfg.task.checkpoint_task = f"{path_checkpoints}/{last_ckpt}"
    model = hydra.utils.instantiate(cfg.task)
    print(f"Testing {task_name}  - {last_ckpt}")
    demo_partial = hydra.utils.instantiate(cfg.test.demo)
    demo = demo_partial(device=device, module=model)

    demo.set_test_params(
        task=task_name,
        sensor=cfg.sensor,
        ckpt=last_ckpt,
        dataset_name=None,
        path_outputs=cfg.test.path_outputs,
        config=cfg,
    )

    demo.init()
    demo.run_model()
    print("*** Demo finished ***")


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    path_outputs = cfg.paths.output_dir
    path_ckpt_encoders = cfg.task.checkpoint_encoder

    path_outputs = resolve_experiment_dir(path_outputs, cfg)
    exp_config = f"{path_outputs}/config.yaml"

    test_cfg = cfg.test.copy()
    data = cfg.data.copy()
    cfg = OmegaConf.load(exp_config)
    cfg.data = data
    cfg.test = test_cfg
    cfg.paths.output_dir = path_outputs
    cfg.task.checkpoint_encoder = path_ckpt_encoders

    demo(cfg)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
