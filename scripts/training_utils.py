import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import json

import string
import os
import random
import subprocess
import torch

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config

def dump_config(data, fname):
    '''
    dump current configuration to an ini file
    :param fname:
    :return:
    '''
    with open(fname, 'w') as file:
        yaml.dump(data, file)
    return fname

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def save_experiment_params(args, experiment_tag, directory):
    t = vars(args)
    params = {k: str(v) for k, v in t.items()}

    git_dir = os.path.dirname(os.path.realpath(__file__))
    git_head_hash = "foo"
    try:
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
    except subprocess.CalledProcessError:
        # Keep the current working directory to move back in a bit
        cwd = os.getcwd()
        os.chdir(git_dir)
        git_head_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).strip()
        os.chdir(cwd)
    params["git-commit"] = str(git_head_hash)
    params["experiment_tag"] = experiment_tag
    for k, v in list(params.items()):
        if v == "":
            params[k] = None
    if hasattr(args, "config_file"):
        config = load_config(args.config_file)
        params.update(config)
    with open(os.path.join(directory, "params.json"), "w") as f:
        json.dump(params, f, indent=4)

def yield_forever(iterator):
    while True:
        for x in iterator:
            yield x

def load_checkpoints(model, optimizer, experiment_directory, args, device):
    if experiment_directory is not None and os.path.exists(experiment_directory):
        model_files = [
            f for f in os.listdir(experiment_directory)
            if f.startswith("model_") and not f.endswith("final")
        ]
    else:
        model_files=[]

    if len(model_files) == 0:
        return
    ids = [int(f[6:]) for f in model_files]
    max_id = max(ids)
    model_path = os.path.join(
        experiment_directory, "model_{:05d}"
    ).format(max_id)
    opt_path = os.path.join(
        experiment_directory, "opt_{:05d}"
    ).format(max_id)
    if not (os.path.exists(model_path) and os.path.exists(opt_path)):
        return

    print("Loading model checkpoint from {}".format(model_path))
    
    try:
        import pdb;pdb.set_trace()
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict)
        print("Loading optimizer checkpoint from {}".format(opt_path))
        optimizer.load_state_dict(
            torch.load(opt_path, map_location=device)
        )
        args.continue_from_epoch = max_id+1
    except:
        model_dict = model.state_dict()
        ckpt_dict = torch.load(model_path, map_location=device)
        main_dict = {k: v for k, v in ckpt_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        mismatch_dict = {k: v for k, v in ckpt_dict.items() if k not in model_dict or v.shape != model_dict[k].shape}
        print(colored(f"Mismatch Dict: {mismatch_dict.keys()}", "red"))
        model_dict.update(main_dict)
        model.load_state_dict(model_dict)
    
        args.continue_from_epoch = 0


def save_checkpoints(epoch, model, optimizer, experiment_directory, is_distributed=False):
    
    if is_distributed:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    torch.save(
        model_state,
        os.path.join(experiment_directory, "model_{:05d}").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_{:05d}").format(epoch)
    )
    
    torch.save(
        model_state,
        os.path.join(experiment_directory, "model_final").format(epoch)
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(experiment_directory, "opt_final").format(epoch)
    )

# pytorch distribute training. 
import torch.distributed as dist
def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()