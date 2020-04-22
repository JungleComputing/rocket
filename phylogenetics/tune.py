import kernel_tuner
import numpy as np
from collections import OrderedDict
import random
import itertools
import sys

compiler_options = ['--expt-extended-lambda', '-std=c++11']

with open("phylogenetics/src/main/cpp/phylogenetics.cu") as f:
    content = f.read()

n = int(2 ** 25)

def is_power_of_two(n):
    x = 1
    while True:
        if x == n:
            return True
        elif x > n:
            return False
        else:
            x *= 2

tune_params = OrderedDict(
        TUNE_USE_SMEM=[0, 1],
        TUNE_THREADS_PER_BLOCK=[i for i in range(128, 1024 + 1, 32)],
        TUNE_ITEMS_PER_THREAD=[i for i in range(1, 20 + 1)],
)

keys = np.unique(np.random.randint(2 ** 31, size=2 * n, dtype=np.int32))
left_keys = np.random.choice(keys, n, replace=False)
right_keys = np.random.choice(keys, n, replace=False)

values = np.ones(n) / n
result = np.float32([0])

args = [
        np.sort(left_keys),
        values,
        np.int32(n),
        np.sort(right_keys),
        values,
        np.int32(n),
        result
]

#keys = list(tune_params.keys())
#values = list(tune_params.values())
#configs = list(itertools.product(*values))
##random.shuffle(configs)
#
#for tune_params in configs:
#    if tune_params[0] and tune_params[1] * tune_params[2] * 8 * 2 > 48 * 1024:
#        continue
#
#    result, env = kernel_tuner.tune_kernel(
#            "tuneCalculateJensenShannon",
#            content,
#            n,
#            args,
#            dict(zip(keys, [[v] for v in tune_params])),
#            compiler_options=compiler_options,
#            verbose=False,
#            quiet=True,
#            lang="C")
#
#    print(tune_params, result[0]['time'], flush=True)
#


configs, env = kernel_tuner.tune_kernel(
        "tuneCalculateCosineSimilarity",
        content,
        n,
        args,
        tune_params,
        verbose=True,
        compiler_options=compiler_options,
        cache="native_tune.json",
        restrictions=["TUNE_USE_SMEM * TUNE_THREADS_PER_BLOCK * TUNE_ITEMS_PER_THREAD * 8 * 2 <= 48 * 1024"],
        strategy='random_sample',
        strategy_options=dict(sample_fraction=0.01),
        lang="C")

for config in configs:
    print(config)
