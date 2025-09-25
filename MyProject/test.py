
# %%===========================================================================

# This script is available in scripts/consult_environments.py
import sinergym
import gymnasium as gym

print(sinergym.__version__)
print(sinergym.ids())

# Make and consult environment
env = gym.make('Eplus-5zone-hot-continuous-stochastic-v1')
print(env.get_wrapper_attr('to_str')())
