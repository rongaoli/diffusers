#!/usr/bin/env python
from argparse import ArgumentParser

from .custom_blocks import CustomBlocksCommand
from .env import EnvironmentCommand
from .fp16_safetensors import FP16SafetensorsCommand

def main():
    parser = ArgumentParser("Diffusers CLI tool", usage="diffusers-cli <command> [<args>]")
    commands_parser = parser.add_subparsers(help="diffusers-cli command helpers")

    # Register commands
    EnvironmentCommand.register_subcommand(commands_parser)
    FP16SafetensorsCommand.register_subcommand(commands_parser)
    CustomBlocksCommand.register_subcommand(commands_parser)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()

if __name__ == "__main__":
    main()
