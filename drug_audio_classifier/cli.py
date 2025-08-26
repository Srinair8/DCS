# cli.py
import argparse
from app import run_cli

parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--test_csv")
args = parser.parse_args()

run_cli(args.audio, args.test_csv)
