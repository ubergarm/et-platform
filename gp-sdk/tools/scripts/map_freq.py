#!/usr/bin/env python3

# Copyright (c) 2025 Ainekko, Co.
# SPDX-License-Identifier: Apache-2.0
# *-------------------------------------------------------------------------
# 
#
# utility remap Chrome Events (aka perfetto) traces into a different device freq
# 
# ./map_freq.py  -i trace.json -o trace.remapped.json -f 1M -t 600M 
#
import json
import re
from pathlib import Path

class ChromeTrace:
    """Faster trace I/O"""
    def __init__(self, fp):
        self.fp = fp
        self.sep = ' '
    def write_header(self):
        print('[', file=self.fp)
    def write_footer(self):
        print(']', file=self.fp)
    def write_event(self, evt):
        print(self.sep, file=self.fp, end='')
        print(json.dumps(evt, separators=(',',':')), file=self.fp)
        self.sep = ','

class Frequency:
    REGEX = re.compile(r'(?P<value>[0-9](.[0-9])?)(?P<prefix>[kmg])?(hz)?', re.IGNORECASE)
    def __init__(self, s):
        match = Frequency.REGEX.match(s)
        if not match:
            raise ValueError('Invalid value for frequency')
        self.value = float(match.group('value'))
        prefix = match.group('prefix')
        if prefix.lower() == 'k':
            self.value *= 1e3
        elif prefix.lower() == 'm':
            self.value *= 1e6
        elif prefix.lower() == 'g':
            self.value *= 1e9

def parse_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=Path, required=True, help='input trace')
    parser.add_argument('-o', '--output', type=Path, required=True, help='output trace')
    parser.add_argument('-f', '--from', dest='from_freq',
            type=Frequency, required=True, help='input frequency')
    parser.add_argument('-t', '--to', dest='to_freq',
            type=Frequency, required=True, help='output frequency')
    parser.add_argument('--tsint', action='store_true',
            help='Round timestamps to nearest integer')
    args = parser.parse_args()
    assert args.input.exists()
    assert args.from_freq.value > 0
    assert args.to_freq.value > 0
    return args

def main():
    args = parse_args()

    ratio = args.from_freq.value / args.to_freq.value

    out = ChromeTrace(args.output.open('w'))
    out.write_header()

    def convert_ts(obj):
        if 'ts' in obj:
            obj['ts'] = obj['ts'] * ratio
            if args.tsint:
                obj['ts'] = int(round(obj['ts']))
        if 'cat' in obj:
            out.write_event(obj)
        else:
            return obj

    with args.input.open() as fp:
        json.load(fp, object_hook=convert_ts)

    out.write_footer()

if __name__ == '__main__':
    main()
