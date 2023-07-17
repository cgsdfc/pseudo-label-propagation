import json
import logging
import pickle
from pathlib import Path as P
from pprint import pformat

import jsons
import matplotlib.pyplot as plt


def save_var(savedir: P, var, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".pkl")
    pickle.dump(var, f.open("wb"))
    logging.info(f"Save Var to {f}")


def load_var(savedir: P, name: str):
    f = savedir.joinpath(name).with_suffix(".pkl")
    assert f.exists(), (savedir, name, f)
    var = pickle.load(f.open("rb"))
    logging.info(f"Load Var from {f}")
    return var


def save_fig(savedir: P, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name)
    assert f.suffix, f"save_fig must have suffix, e.g., .png/.eps {name}"
    plt.tight_layout()
    plt.savefig(str(f))
    plt.close()
    logging.info(f"Save Fig to {f}")


def save_json(savedir: P, var, name: str):
    savedir.mkdir(exist_ok=1, parents=1)
    f = savedir.joinpath(name).with_suffix(".json")
    f.write_text(jsons.dumps(var, jdkwargs=dict(indent=4)))
    logging.info(f"Save Var to {f}")


def load_json(savedir: P, name: str):
    f = savedir.joinpath(name).with_suffix(".json")
    assert f.exists(), (savedir, name, f)
    var = json.load(f.open())
    logging.info(f"Load Var from {f}")
    return var


def train_begin(savedir: P, config: dict, message: str = None):
    message = message or "Train begins"
    logging.info(f"{message} {pformat(config)}")
    save_json(savedir, config, "config")


def train_end(savedir: P, metrics: dict, message: str = None):
    message = message or "Train ends"
    logging.info(f"{message} {metrics}")
    save_json(savedir, metrics, "metrics")
