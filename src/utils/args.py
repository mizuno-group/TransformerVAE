import os, sys
import argparse
from collections import defaultdict

import yaml
from addict import Dict

def update_with_check(origin, after, path, modifieds, filename):
    if isinstance(origin, dict) and isinstance(after, dict):
        for key, value in after.items():
            if key not in origin:
                origin[key] = value
            else:
                origin[key], modifieds = update_with_check(
                    origin[key], value, path+(key,), modifieds, filename)
        return origin, modifieds
    else:
        if origin != after:
            if origin != {}:
                for mpath in modifieds.keys():
                    if path[:len(mpath)] == mpath or mpath[:len(path)] == path:
                        print(f"WARNING: {'.'.join(path)} was overwritten for multiple times:", 
                            file=sys.stderr)
                        for filename0 in modifieds[mpath]:
                            print("  ", filename0, file=sys.stderr)
                        print("  ", filename, file=sys.stderr)
                        
                modifieds[path].append(filename)
        return after, modifieds

def gather_args(config):
    args = []
    if isinstance(config, dict):
        if 'argname' in config:
            arg_args = {}
            for key, value in config.items():
                if key == 'argname':
                    continue
                elif key == 'type':
                    arg_args['type'] = eval(config.type)
                else:
                    arg_args[key] = value
            args = [(config.argname, arg_args)]
        else:
            for child in config.values():
                args += gather_args(child)
    elif isinstance(config, list):
        for child in config:
            args += gather_args(child)
    return args

def fill_args(config, args):
    if isinstance(config, dict):
        if 'argname' in config:
            return args[config.argname]
        else:
            for label, child in config.items():
                config[label] = fill_args(child, args)
            return config
    elif isinstance(config, list):
        config = [fill_args(child, args) for child in config]
        return config
    else:
        return config

def subs_vars(config, vars):
    if isinstance(config, str):
        if config in vars:
            return vars[config]
        for key, value in vars.items():
            config = config.replace(key, str(value))
        return config
    elif isinstance(config, dict):
        return Dict({label: subs_vars(child, vars) for label, child in config.items()})
    elif isinstance(config, list):
        return [subs_vars(child, vars) for child in config]
    else:
        return config

def delete_args(config):
    if isinstance(config, dict):
        new_config = Dict()
        for key, value in config.items():
            if value == '$delete':
                continue
            elif isinstance(value, (dict, list)):
                value = delete_args(value)
            new_config[key] = value
        return new_config
    elif isinstance(config, list):
        return [delete_args(child) for child in config]
    else:
        return config

def load_config(config_dir, default_configs):
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs='*', default=[])
    args = parser.parse_known_args(argv)[0]
    config = None
    modifieds = defaultdict(list)
    for file in default_configs+args.config:
        if os.path.exists(file):
            with open(file, 'rb') as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))
        else:
            with open(os.path.join(config_dir, file)+".yaml", 'rb') as f:
                aconfig = Dict(yaml.load(f, yaml.Loader))
        # config.update(aconfig)
        if config is None: 
            config = aconfig
        else:
            config, modifieds = update_with_check(config, aconfig, tuple(), modifieds, file)
    config = Dict(config)
    args = gather_args(config)
    for arg_name, arg_args in args:
        parser.add_argument(f"--{arg_name}", **arg_args)
    args = vars(parser.parse_args(argv))
    config = fill_args(config, args)
    variables = {}
    if 'variables' in config.keys():
        variables.update(config['variables'])
    config = subs_vars(config, variables)
    config = delete_args(config)
    return Dict(config)