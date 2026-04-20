#! /usr/bin/python3

# Copyright 2026 Adam Scott Thomas
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import pickle
from collections import defaultdict

from time_cell import TimeCell, Result, Config

f_name = "rule_sweep.p"
data = []
with open(f_name, "rb") as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

print(len(data))

stats = defaultdict(lambda: defaultdict(int))
for config, result in data:
    stats[config.rule]["count"] += 1
    if result is None:
        stats[config.rule]["nones"] += 1
    else:
        stats[config.rule]["total_cycle"] += result.cycle_length

for r, v in stats.items():
    print("Rule: {:3d} Count: {:3d} Nones: {:3d} Total: {:5d}".format(r, v["count"], v["nones"], v["total_cycle"]))

##### keepers for good diversity: [30, 45, 73, 97, 110, 137, 161, 165, 169]

# no time travel: 161, 126, 165,    135,    73,    109,    18,   149,   86,    22, 151, 183, 90, 30
#                tri, tri, weird, weird, square, square, weird, werid, weird, tri, tri, tri, cool, cool,

# medium time travel: 129, 135, 137, 18, 146, 147, 149, 22, 150, 151, 153, 30, 161,
#  37, 165, 41, 169, 45, 54, 182, 183, 60, 193, 195, 73, 75, 86, 89, 90, 91, 97, 101, 102, 105, 107, 109, 110, 120, 121, 122, 124, 126

# long time travel: 161, 135, 73, 109, 18, 149, 86, 22, 151, 183, 30
