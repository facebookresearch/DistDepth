# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from execute_func import Trainer
from options import DistDepthOptions

options = DistDepthOptions()
opts = options.parse()

if __name__ == "__main__":
    trainer = Trainer(opts)
    if opts.exe == 'eval_save':
        trainer.eval_save()
    elif opts.exe == 'eval_measure':
        trainer.eval_measure()
    else:
        raise NotImplementedError("choose valid execution, see options.py")