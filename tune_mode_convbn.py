#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#


# Import required libraries
from typing import Tuple
from functools import partial
from operator import attrgetter

import torch
import torch.nn as nn
import torch.fx as fx
from torch.nn.utils.fusion import fuse_conv_bn_weights


# Function to compute parameters of the convolution layer on-the-fly
def compute_params_on_the_fly(conv_weight, conv_bias, bn_weight, bn_bias, weight_coeff, bias_delta):
    weight_on_the_fly = conv_weight 
    bias_on_the_fly = conv_bias
    coefff_on_the_fly = weight_coeff * bn_weight.view_as(weight_coeff) # shape of [C_out, 1, 1, 1]
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly # shape of [C_out, C_in, k, k] in Conv2d
    bias_on_the_fly = (bias_on_the_fly + bias_delta) * coefff_on_the_fly.flatten() + bn_bias # shape of [C_out]
    return weight_on_the_fly, bias_on_the_fly

# Function to get the on-the-fly parameters
def get_params(self):
    weight_on_the_fly, bias_on_the_fly = compute_params_on_the_fly(
        self.weight, # shape of [C_out, C_in, k, k] in Conv2d
        self.bias if self.bias is not None else torch.zeros_like(self.bn_bias), # shape of [C_out] in Conv2d
        self.bn_weight,
        self.bn_bias,
        self.weight_coeff,
        self.bias_delta
    )
    return weight_on_the_fly, bias_on_the_fly

# Function to perform modified forward pass with on-the-fly parameters
def modified_forward(self, input):
    weight_on_the_fly, bias_on_the_fly = get_params(self)
    return self.__class__._conv_forward(self, input, weight_on_the_fly, bias_on_the_fly)

# Function to turn on the tune mode by fusing convolution and batch normalization layers
def turn_on_tune_mode(conv, bn):
    with torch.no_grad():
        weight_coeff = torch.rsqrt(bn.running_var + bn.eps) # shape of [C_out] in Conv2d
        weight_coeff = torch.tensor(weight_coeff.reshape([-1] + [1] * (len(conv.weight.shape) - 1))) # shape of [C_out, 1, 1, 1] in Conv2d
        conv.register_buffer('weight_coeff', weight_coeff)
        conv.register_buffer('bias_delta', - bn.running_mean) # shape of [C_out] in Conv2d
        conv.bn_weight = bn.weight
        conv.bn_bias = bn.bias
        del bn.weight
        del bn.bias
    conv.forward = partial(modified_forward, conv)

# Function to turn on the deploy mode by fusing convolution and batch normalization layers
def turn_on_deploy_mode(conv, bn):
    with torch.no_grad():
        new_weight, new_bias = fuse_conv_bn_weights(conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)
        conv.weight = new_weight
        conv.bias = new_bias

        del bn.weight
        del bn.bias


# Helper function to split a qualname into parent path and last atom.
def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name


# Main function to turn on the optimization mode (tune or deploy) for the given model
def turn_on(model: torch.nn.Module, mode = 'Tune') -> torch.nn.Module:
    """
    model: the Module to optimize (bn modules are reserved in a module list in model.reseverd_bns)
    mode: tune or deploy
    """    

    mode = mode.lower()
    assert mode in ['tune', 'deploy']

    # Symbolically trace the input model to create an FX GraphModule
    fx_model: fx.GraphModule = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    model.reserved_bns = nn.ModuleList()

    patterns = [(torch.nn.Conv1d, torch.nn.BatchNorm1d),
                (torch.nn.Conv2d, torch.nn.BatchNorm2d),
                (torch.nn.Conv3d, torch.nn.BatchNorm3d)]

    # Iterate through nodes in the graph to find ConvBN blocks
    for node in fx_model.graph.nodes:
        if node.op != 'call_module': # If our current node isn't calling a Module then we can ignore it.
            continue
        found_pair = [node for conv_class, bn_class in patterns if type(modules[node.target]) is bn_class and type(modules[node.args[0].target]) is conv_class]
        if not found_pair or len(node.args[0].users) > 1: # Not a conv-BN pattern or output of conv is used by other nodes
            continue

        # Find a pair of conv and bn to optimize
        conv_name = node.args[0].target
        bn_name = node.target

        print(f'Turn on mode {mode} for {conv_name} and {bn_name}')
        conv = modules[conv_name]
        bn = modules[bn_name]
        
        # Turn on the optimization mode and move the bn to reserved_bns
        if mode == 'tune':
            turn_on_tune_mode(conv, bn)
        else:
            turn_on_deploy_mode(conv, bn)

        model.reserved_bns.append(bn)

        # Remove the original bn from the model
        parent_name, name = _parent_name(bn_name)
        if parent_name != '':
            getter = attrgetter(parent_name)
            bn_parent = getter(model)
        else:
            bn_parent = model
        setattr(bn_parent, name, nn.Identity())
