import hashlib
from typing import Dict, List, Optional
import random
import string
import neptune

__init_params = None
__run_id: str = "run_"


def init_logger(init_args):
    global __init_params
    global __run_id
    __init_params = init_args
    __run_id += ''.join(random.choice(string.digits + string.ascii_letters) for _ in range(6))
    neptune.init('rashchedrin/Proverbot9001',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiY2NiNTk1NWItOTQ0MS00ODdhLWE3N2MtMjE4ZDY2OGQ2MmZmIn0=")


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


def log_metric(metric_name: str, metric_value):
    if isinstance(metric_value, str):
        neptune.log_text(metric_name, metric_value)
    else:
        neptune.log_metric(metric_name, metric_value)


def log_image(pic_name, filename):
    neptune.log_image(pic_name, filename)


def log_metrics(metrics: Dict):
    for metric_name, metric_value in metrics.items():
        log_metric(metric_name, metric_value)


def set_experiment_as_lemma(module_name, lemma_name, args):
    module_and_lemma_name = "M_" + str(module_name) + "_L_" + str(lemma_name)
    experiment_name = "V0.1_" + args.traverse_method + "_" + module_and_lemma_name
    experiment_params = {k: str(v) for k, v in vars(args).items()}
    experiment_params.update({
        "module_and_lemma_name": module_and_lemma_name
    })
    set_experiment(experiment_name, experiment_params, tags=[args.experiment_tag])
