import neptune
from typing import Dict, List, Optional

__init_params = None


def init_logger(init_args):
    global __init_params
    __init_params = init_args
    neptune.init('rashchedrin/Proverbot9001',
                 api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiY2NiNTk1NWItOTQ0MS00ODdhLWE3N2MtMjE4ZDY2OGQ2MmZmIn0=")


def set_experiment(name: str, experiment_parameters: Dict[str, str], tags: Optional[List] = None):
    global __init_params
    all_params = {}
    all_params.update(__init_params)
    all_params.update(experiment_parameters)
    neptune.create_experiment(name=name, params=experiment_parameters, tags=tags)


def log_metric(metric_name: str, metric_value):
    if isinstance(metric_value, str):
        neptune.log_text(metric_name, metric_value)
    else:
        neptune.log_metric(metric_name, metric_value)


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
    set_experiment(experiment_name, experiment_params, tags=['V_01'])
