import hashlib
from typing import Dict, List, Optional
import random
import string
import neptune
import pickle
import os

__init_params = None
__run_id: str = "run_"
__git_patch_filename: str = ""


def init_logger(init_args):
    global __init_params
    global __run_id
    global __git_patch_filename
    __init_params = init_args
    __run_id += ''.join(random.choice(string.digits + string.ascii_letters) for _ in range(6))
    neptune.init('rashchedrin/Proverbot9001',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiY2NiNTk1NWItOTQ0MS00ODdhLWE3N2MtMjE4ZDY2OGQ2MmZmIn0=")
    __git_patch_filename = "run_" + __run_id + ".git_diff"
    os.system("git diff > " + __git_patch_filename)

def run_id():
    return __run_id

def dict_digest(d: Dict):
    string = str(sorted(list(d.items()), key=lambda x: x[0]))
    return hashlib.sha224(bytes(string, 'utf-8')).hexdigest()


def set_experiment(name: str, experiment_parameters: Dict[str, str], tags: Optional[List] = None):
    global __init_params
    global __run_id
    all_params = {}
    all_params.update(__init_params)
    all_params.update(experiment_parameters)
    digest_params = all_params.copy()
    del digest_params['module_and_lemma_name']
    params_digest = dict_digest(digest_params)[:10]
    tags = [params_digest] if tags is None else tags + [params_digest]
    tags += [__run_id]
    neptune.create_experiment(name=name, params=all_params, tags=tags)
    neptune.log_artifact(__git_patch_filename)


def _append_to_file(filename, key, what):
    data = (key, what)
    with open(filename, "ab") as f:
        pickle.dump(data, f)


def _read_from_file(filename):
    data = []
    with open(filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data


def log_metric(filename, metric_name: str, metric_value):
    _append_to_file(filename, metric_name, metric_value)
    if isinstance(metric_value, str):
        neptune.log_text(metric_name, metric_value)
    else:
        neptune.log_metric(metric_name, metric_value)


def log_image(pic_name, filename):
    neptune.log_image(pic_name, filename)


def log_metrics(filename, metrics: Dict):
    for metric_name, metric_value in metrics.items():
        log_metric(filename, metric_name, metric_value)


def set_experiment_as_lemma(module_name, lemma_name, args):
    module_and_lemma_name = "M_" + str(module_name) + "_L_" + str(lemma_name)
    experiment_name = "V0.1_" + args.traverse_method + "_" + module_and_lemma_name
    experiment_params = {k: str(v) for k, v in vars(args).items()}
    experiment_params.update({
        "module_and_lemma_name": module_and_lemma_name
    })
    set_experiment(experiment_name, experiment_params, tags=[args.experiment_tag])
