# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

from execute_func import Trainer
from execute_func_multi import Trainer_multi
from options import DistDepthOptions

options = DistDepthOptions()
opts = options.parse()

if __name__ == "__main__":
    
    if opts.exe == 'train':
        trainer = Trainer(opts)
        trainer.train()
    if opts.exe == 'eval_save':
        trainer = Trainer(opts)
        trainer.eval_save()
    elif opts.exe == 'eval_save_all':
        trainer = Trainer(opts)
        trainer.eval_save()
    elif opts.exe == 'eval_measure':
        trainer = Trainer(opts)
        trainer.eval_measure()
    elif opts.exe == 'eval_measure-M':
        trainer_multi = Trainer_multi(opts)
        trainer_multi.eval_measure_multi()
    else:
        raise NotImplementedError("choose valid execution, see options.py")