# Copyright 2020 Magic Leap, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#  Originating Author: Zak Murez (zak.murez.com)

#import pytorch_lightning as pl
import lightning.pytorch as pl

import torch

# FIXME: should not be necessary, but something is remaining
# in memory between train and val
class CudaClearCacheCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        #print("***CACHE_train_start")
        torch.cuda.empty_cache()
    def on_validation_start(self, trainer, pl_module):
        #print("***CACHE_val_start")
        torch.cuda.empty_cache()
    def on_validation_end(self, trainer, pl_module):
        #print("***CACHE_val_end")
        torch.cuda.empty_cache()