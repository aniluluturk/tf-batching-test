import tfhelpers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", help="batch size")
parser.add_argument("-n", help="number of instances")
parser.add_argument("-r", help="randomize dataset for batches", action="store_true")
args = parser.parse_args()
if args.r:
    randomize=True
else:
    randomize=False
print randomize
if args.s is not None and args.n is not None:
    tfhelpers.create_partition("/tmp/", int(args.s), int(args.n), randomize)
else:
    tfhelpers.create_partition("/tmp/", randomize)

tfhelpers.create_test_files("/tmp/")
