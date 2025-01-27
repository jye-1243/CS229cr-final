# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass

from foundations import desc
from lottery.desc import LeverageDesc


def make_BranchDesc(BranchHparams: type, name: str):
    @dataclass
    class BranchDesc(desc.Desc):
        lottery_desc: LeverageDesc
        branch_hparams: BranchHparams

        @staticmethod
        def name_prefix(): return 'lottery_branch_' + name

        @staticmethod
        def add_args(parser: argparse.ArgumentParser, defaults: LeverageDesc = None):
            LeverageDesc.add_args(parser, defaults)
            BranchHparams.add_args(parser)

        @classmethod
        def create_from_args(cls, args: argparse.Namespace):
            return BranchDesc(LeverageDesc.create_from_args(args), BranchHparams.create_from_args(args))

    return BranchDesc
