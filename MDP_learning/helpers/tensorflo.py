import os
import argparse
import re
import time
from os.path import join, getsize
from tensorboard.backend.event_processing import \
    plugin_event_accumulator as event_accumulator
import MDP_learning.helpers.extract_scalars as es


def check(f, args_parsed):
    now = int(time.time())
    rx = re.compile(args_parsed.regex)
    return rx.search(f) and \
           os.path.getsize(f) >= args_parsed.minsz and \
           (now - int(os.path.getmtime(f))) / (60 * 60) < args_parsed.latest


parser = argparse.ArgumentParser(description='Process some logs.', epilog="And that's how you'd foo a bar")
# parser.add_argument('--mode', choices=['train', 'test'], default='test')
parser.add_argument('--logdir', type=str, default='./',
                    help='foldername')
parser.add_argument('--outdir', type=str, default='./csv',
                    help='output foldername')
parser.add_argument('--regex', type=str, default='',
                    help='regex matching')
parser.add_argument('--minsz', type=int, default=0,
                    help='minimum file size')
parser.add_argument('--latest', type=int, default=1000,
                    help='only consider files newer than this number of hours')
parser.add_argument('-d', '--dry', action='store_true',
                    help='Do a dry run, nothing is written')
parser.add_argument('-l', '--list', action='store_true',
                    help='List EventAccumulator tags')
parser.add_argument('-f', '--flat', action='store_true',
                    help='Do not recreate folder structure in outdir')
args = parser.parse_args()
print(args)

if args.list:
    mf = []
    for root, dirs, files in os.walk(args.logdir):
        for file in files:
            if file.startswith("events.out.tfevents."):
                f = os.path.join(root, file)
                if check(f, args):
                    mf.extend(f)

                    event_acc = event_accumulator.EventAccumulator(f)
                    event_acc.Reload()

                    # Print tags of contained entities, use these names to retrieve entities as below
                    print('========================================================================')
                    print(f)
                    print(event_acc.Tags())

# es.run(args.logdir)

multiplexer = es.create_multiplexer(args.logdir)
print('Running in: '.format(args.logdir))
for run_name, data in multiplexer.Runs().items():
    for tag_name in data['tensors']:
        accu = multiplexer.GetAccumulator(run_name)
        if check(accu.path, args):
            output_filename = '%s___%s.csv' % (es.munge_filename(run_name), es.munge_filename(tag_name))
            path = accu.path.strip('..')
            output_dir = '{}/{}'.format(args.outdir, '' if args.flat else path)
            output_dir = os.path.normpath(output_dir)
            output_filepath = os.path.join(output_dir, output_filename)
            print("Exporting (run=%r, tag=%r) to %r..." % (run_name, tag_name, output_filepath))
            if not args.dry:
                es.mkdir_p(output_dir)
                es.export_scalars(multiplexer, run_name, tag_name, output_filepath)
