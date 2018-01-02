import numpy as np
import fancyimpute

print("Loading memory")
mem = np.load('/home/aocc/code/DL/MDP_learning/save_memory/first_20mill/BipedalWalker-v2CORRUPT0.1.npy')
print("Imputing memory")
memory_final = fancyimpute.SoftImpute().complete(mem)
print("Saving imputed memory")
np.save('/home/aocc/code/DL/MDP_learning/save_memory/first_20mill/BipedalWalker-v2IMPUTED.1', memory_final)
