"""Script used to train a MIME."""
import argparse
import logging
import os
import sys
from termcolor import colored

import numpy as np
import torch
import time

PREFETCH_DATALOADER = False # ! Warning: this leads to generate many multiple threads for cpu recursively.
if PREFETCH_DATALOADER:
    from torch.utils.data import DataLoader
    from prefetch_generator import BackgroundGenerator
    from torch.utils.data import DataLoader as Ori_DataLoader
    class DataLoader(Ori_DataLoader):
        def __iter__(self):
            return BackgroundGenerator(super().__iter__())
else:
    from torch.utils.data import DataLoader


from training_utils import (id_generator, save_experiment_params, \
    load_config, yield_forever, load_checkpoints, save_checkpoints, synchronize)
from scene_synthesis.datasets import get_encoded_dataset, filter_function
from scene_synthesis.networks import build_network, optimizer_factory
from scene_synthesis.stats_logger import StatsLogger, WandB
from main_utils import train_get_parser
import torch.nn as nn

def main(argv):
    
    args = train_get_parser(argv)

    is_distributed = args.ngpu > 1 and args.distributed

    if is_distributed:
        print('start distributed ************\n')
        local_rank = 0
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        
    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    
    
    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Create an experiment directory using the experiment_tag
    if args.experiment_tag is None:
        experiment_tag = id_generator(9)
    else:
        experiment_tag = args.experiment_tag

    experiment_directory = os.path.join(
        args.output_directory,
        experiment_tag
    )
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)

    if (not is_distributed) or (dist.get_rank() == 0): # ! create dir, save data, or print
        # Save the parameters of this run to a file
        save_experiment_params(args, experiment_tag, experiment_directory)
        print("Save experiment statistics in {}".format(experiment_directory))

    # Parse the config file
    config = load_config(args.config_file)

    train_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        path_to_bounds=None,
        augmentations=config["data"].get("augmentations", None),
        split=config["training"].get("splits", ["train", "val"])
    )
    # Compute the bounds for this experiment, save them to a file in the
    # experiment directory and pass them to the validation dataset
    path_to_bounds = os.path.join(experiment_directory, "bounds.npz")
    print("Saved the dataset bounds in {}".format(path_to_bounds))

    if (not is_distributed) or (dist.get_rank() == 0): # ! create dir, save data, or print
        np.savez(
            path_to_bounds,
            sizes=train_dataset.bounds["sizes"],
            translations=train_dataset.bounds["translations"],
            angles=train_dataset.bounds["angles"]
        )
    
    validation_dataset = get_encoded_dataset(
        config["data"],
        filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        path_to_bounds=path_to_bounds,
        augmentations=None,
        split=config["validation"].get("splits", ["test"])
    )
    
    print("Loaded {} training scenes with {} object types".format(
        len(train_dataset), train_dataset.n_object_types)
    )
    # print("Training set has {} bounds".format(train_dataset.bounds))
    
    print("Loaded {} validation scenes with {} object types".format(
        len(validation_dataset), validation_dataset.n_object_types)
    )
    
    ### convert data loader to dist
    if is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, 
                                    num_replicas=dist.get_world_size(),
                                    rank=dist.get_rank(),
                                    shuffle=True)
        val_sampler = torch.utils.data.DistributedSampler(validation_dataset, 
                                    num_replicas=dist.get_world_size(),
                                    rank=dist.get_rank(),
                                    shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"].get("batch_size", 128),
            sampler=train_sampler,
            num_workers=args.n_processes,
            collate_fn=train_dataset.collate_fn,
            drop_last=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=config["validation"].get("batch_size", 1),
            sampler=val_sampler,
            num_workers=args.n_processes,
            collate_fn=validation_dataset.collate_fn,
            drop_last=True,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"].get("batch_size", 128),
            num_workers=args.n_processes,
            collate_fn=train_dataset.collate_fn,
            shuffle=True
        )
        val_loader = DataLoader(
            validation_dataset,
            batch_size=config["validation"].get("batch_size", 1),
            num_workers=args.n_processes,
            collate_fn=validation_dataset.collate_fn,
            shuffle=False
        )

    # Make sure that the train_dataset and the validation_dataset have the same
    # number of object categories
    assert train_dataset.object_types == validation_dataset.object_types

    # Build the network architecture to be used for training
    network, train_on_batch, validate_on_batch = build_network(
        train_dataset.feature_size, train_dataset.n_classes,
        config, args.weight_file, args.weight_strict, device=device
    )

    # Build an optimizer object to compute the gradients of the parameters
    optimizer = optimizer_factory(config["training"], network.parameters())
    
    # Load the checkpoints if they exist in the experiment directory
    load_checkpoints(network, optimizer, args.load_ckpt_dir, args, device)
    
    ### convert model to dist
    if is_distributed:
        print("Dist Train, Let's use", torch.cuda.device_count(), "GPUs!")
        network = torch.nn.parallel.DistributedDataParallel(
            network, device_ids=[local_rank], output_device=local_rank,
            # find_unused_parameters=False,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    elif args.ngpu > 1:
        if torch.cuda.is_available():
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            network = nn.DataParallel(network)

    if (not is_distributed) or (dist.get_rank() == 0):
        if args.with_wandb_logger:
            os.environ["WANDB_API_KEY"] = '957dd25a6e0eb03475e640789ecc6c0ab95745c8'
            os.environ["WANDB_MODE"] = "offline"

            WandB.instance().init(
                config,
                model=network,
                project=config["logger"].get(
                    "project", "autoregressive_transformer"
                ),
                name=experiment_tag,
                watch=False,
                log_frequency=10
            )

        # Log the stats to a file
        StatsLogger.instance().add_output_file(open(
            os.path.join(experiment_directory, "stats.txt"),
            "w"
        ))

    epochs = config["training"].get("epochs", 150)
    steps_per_epoch = config["training"].get("steps_per_epoch", 500)
    save_every = config["training"].get("save_frequency", 10)
    val_every = config["validation"].get("frequency", 100)

    # Do the training
    best_val_loss = 1e10
    for i in range(args.continue_from_epoch, epochs):
        network.train()
        train_cost_time = 0
        data_time_epoch = 0
        data_start_time = time.time()
        for b, sample in zip(range(steps_per_epoch), yield_forever(train_loader)):
            # Move everything to device
            for k, v in sample.items():
                sample[k] = v.to(device)
            start_time = time.time()
            data_cost_time = start_time-data_start_time
            data_time_epoch += data_cost_time
            # add dataset normalization information during training.
            batch_loss = train_on_batch(network, optimizer, sample, config, train_dataset)
            cost_time = time.time() - start_time
            train_cost_time += cost_time
            data_start_time = time.time()
            if (not is_distributed) or (dist.get_rank() == 0):
                StatsLogger.instance().print_progress(i+1, b+1, batch_loss, {'inference':cost_time, 'data':data_cost_time})
        
        if (not is_distributed) or (dist.get_rank() == 0):
            if (i % save_every) == 0:
                tmp_flag = args.ngpu > 1
                save_checkpoints(
                    i,
                    network,
                    optimizer,
                    experiment_directory,
                    is_distributed=tmp_flag,
                )
            StatsLogger.instance().clear()

        if i % val_every == 0 and i > 0:
            if (not is_distributed) or (dist.get_rank() == 0):
                print("====> Validation Epoch ====>")

            network.eval()
            for b, sample in enumerate(val_loader):
                # Move everything to device
                for k, v in sample.items():
                    sample[k] = v.to(device)
                batch_loss = validate_on_batch(network, sample, config)

                if (not is_distributed) or (dist.get_rank() == 0):
                    StatsLogger.instance().print_progress(-1, b+1, batch_loss)
                    if batch_loss < best_val_loss:
                        best_val_loss = batch_loss
                        if args.ngpu > 1:
                            save_state = network.module.state_dict()
                        else:
                            save_state = network.state_dict()
                        torch.save(
                            save_state,
                            os.path.join(experiment_directory, "model_best").format(i)
                        )
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(experiment_directory, "opt_best").format(i)
                        )
            if (not is_distributed) or (dist.get_rank() == 0):
                StatsLogger.instance().clear()
                print("====> Validation Epoch ====>")

if __name__ == "__main__":
    main(sys.argv[1:])
