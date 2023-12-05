from argparse import ArgumentParser
import json
import os
import random

from configs.config import Config
from misc.logger import get_logger
from misc.global_counter import init_counter
from datasets.gqa import run_gqa


def set_seed(seed: int = 42) -> None:
    """set a seed integer for random sampling algorithms

    Args:
        seed (int, optional): Defaults to 42.
    """
    random.seed(seed)


def get_args_parser() -> ArgumentParser:
    """define arguments

    Returns:
        argparse.ArgumentParser:
    """
    parser = ArgumentParser("", add_help=False)
    parser.add_argument(
        "--config",
        metavar="PATH",
        default="./configs/config_default.json",
        help="path to config",
    )
    return parser


def load_json(path: str) -> dict:
    """wrapper to load json files

    Args:
        path (str)

    Returns:
        dict
    """
    return json.load(open(path))


def main(args: ArgumentParser) -> None:
    """loads data and triggers generation for all dataset splits

    Args:
        args (ArgumentParser):
    """
    cfg = Config(args.config)
    logger = get_logger(cfg)
    logger.info(cfg)
    cpt_counter = init_counter()
    set_seed(seed=42)

    # load data
    logger.info("load data")
    train_set = load_json(cfg.get("gqa_sg_train"))
    valid_set = load_json(cfg.get("gqa_sg_valid"))

    if ("val" in cfg.get("generate_captions")) or (
        "val_subset" in cfg.get("generate_captions")
    ):
        logger.info("generate validation captions")
        # run dataset creation for val split
        run_subset = True if "val_subset" in cfg.get("generate_captions") else False
        valid_results = run_gqa(
            valid_set,
            cfg,
            cpt_counter,
            logger,
            cfg.get("filter_noisy"),
            cfg.get("relaxed_mode"),
            run_subset,
        )
        if cfg.get("save_results"):
            file_name = cfg.get("output_filename")
            os.makedirs("./results", exist_ok=True)
            logger.info(f"save validation captions: ./results/{file_name}_val.json")
            with open(f"./results/{file_name}_val.json", "w") as outfile:
                json.dump(valid_results, outfile)

    if "train" in cfg.get("generate_captions"):
        logger.info("generate training captions")
        # run dataset creation for train split
        train_results = run_gqa(
            train_set,
            cfg,
            cpt_counter,
            logger,
            cfg.get("filter_noisy"),
            cfg.get("relaxed_mode"),
        )
        if cfg.get("save_results"):
            file_name = cfg.get("output_filename")
            os.makedirs("./results", exist_ok=True)
            logger.info(f"save training captions: ./results/{file_name}_train.json")
            with open(f"./results/{file_name}_train.json", "w") as outfile:
                json.dump(train_results, outfile)

    logger.info("generation process completed")


if __name__ == "__main__":
    parser = ArgumentParser(
        "Vision and language mismatch detection", parents=[get_args_parser()]
    )

    args = parser.parse_args()
    main(args)
