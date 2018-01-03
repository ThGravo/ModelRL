import tensorflow as tf
import pandas as pd
from tensorboard.backend.event_processing import \
    plugin_event_multiplexer as event_multiplexer
from tensorboard.backend.event_processing import \
    plugin_event_accumulator as event_accumulator



event_acc = event_accumulator.EventAccumulator(args.logdir)
event_acc.Reload()

# Print tags of contained entities, use these names to retrieve entities as below
print(event_acc.Tags())

# E. g. get all values and steps of a scalar called 'l2_loss'
xy_l2_loss = [(s.step, s.value) for s in event_acc.Scalars('l2_loss')]

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))

pd.DataFrame(event_acc.Scalars('Loss')).to_csv('Loss.csv')
