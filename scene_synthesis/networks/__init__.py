from functools import partial
import torch
try:
    from radam import RAdam
except ImportError:
    pass

from .autoregressive_transformer import AutoregressiveTransformer, \
    AutoregressiveTransformerEncodePredictHumanOneHot, \
    train_on_batch as train_on_batch_simple_autoregressive, \
    validate_on_batch as validate_on_batch_simple_autoregressive

from .hidden_to_output import AutoregressiveDMLL, get_bbox_output
from .feature_extractors import get_feature_extractor


def hidden2output_layer(config, n_classes):
    config_n = config["network"]
    hidden2output_layer = config_n.get("hidden2output_layer")

    if hidden2output_layer == "autoregressive_mlc":
        return AutoregressiveDMLL(
            config_n.get("hidden_dims", 768),
            n_classes,
            config_n.get("n_mixtures", 4),
            get_bbox_output(config_n.get("bbox_output", "autoregressive_mlc")),
            config_n.get("with_extra_fc", False),
        )
    else:
        raise NotImplementedError()


def optimizer_factory(config, parameters):
    """Based on the provided config create the suitable optimizer."""
    optimizer = config.get("optimizer", "Adam")
    lr = config.get("lr", 1e-3)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer == "SGD":
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer == "RAdam":
        return RAdam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError()


def build_network(
    input_dims,
    n_classes,
    config,
    weight_file=None,
    weight_strict=True,
    device="cpu"):
    network_type = config["network"]["type"]

    if 'contact_as_query' in config['network'].keys() and config['network']['contact_as_query']:
        print('run human contact network.')
        n_classes += 4

    
    if network_type == "autoregressive_transformer":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformer(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    
    elif network_type == "autoregressive_transformer_human_anchor_one_hot":
        train_on_batch = train_on_batch_simple_autoregressive
        validate_on_batch = validate_on_batch_simple_autoregressive
        network = AutoregressiveTransformerEncodePredictHumanOneHot(
            input_dims,
            hidden2output_layer(config, n_classes),
            get_feature_extractor(
                config["feature_extractor"].get("name", "resnet18"),
                freeze_bn=config["feature_extractor"].get("freeze_bn", True),
                input_channels=config["feature_extractor"].get("input_channels", 1),
                feature_size=config["feature_extractor"].get("feature_size", 256),
            ),
            config["network"]
        )
    else:
        raise NotImplementedError()

    # Check whether there is a weight file provided to continue training from
    if weight_file is not None and weight_file != "None":
        print("Loading weight file from {}".format(weight_file))
        
        module_dict = torch.load(weight_file, map_location=device)
        all_keys = list(module_dict.keys())
        if 'module' in all_keys[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in module_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        else:
            new_state_dict = module_dict
        
        print('weight: ', weight_strict)
        if not weight_strict:
            for k, v in new_state_dict.items():
                if k not in network.state_dict() or v.shape != network.state_dict()[k].shape:
                    print("{} is not in the model".format(k))
                    continue
                network.state_dict()[k].copy_(v)

        else:
            network.load_state_dict(
                new_state_dict,
                strict=weight_strict,
            )
        
    network.to(device)
    return network, train_on_batch, validate_on_batch
