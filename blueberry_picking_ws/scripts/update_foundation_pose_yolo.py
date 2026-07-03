#!/usr/bin/env python3
"""Update yolo_model and yolo_conf_threshold in foundation_pose.yaml."""

from __future__ import annotations

import argparse
import os
import re


def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    default_yaml = os.path.join(
        root, 'src', 'picking_perception', 'config', 'foundation_pose.yaml')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yaml', default=default_yaml)
    parser.add_argument('--model', required=True, help='Absolute path to best.pt')
    parser.add_argument('--conf', type=float, default=0.38)
    args = parser.parse_args()

    model = os.path.abspath(args.model)
    if not os.path.isfile(model):
        raise SystemExit(f'Model not found: {model}')
    if not os.path.isfile(args.yaml):
        raise SystemExit(f'YAML not found: {args.yaml}')

    with open(args.yaml, encoding='utf-8') as fh:
        text = fh.read()

    text, n_model = re.subn(
        r'^(\s*yolo_model:\s*).*$',
        rf'\1{model}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text, n_conf = re.subn(
        r'^(\s*yolo_conf_threshold:\s*).*$',
        rf'\g<1>{args.conf}',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if n_model == 0 or n_conf == 0:
        raise SystemExit('Could not find yolo_model or yolo_conf_threshold in yaml')

    with open(args.yaml, 'w', encoding='utf-8') as fh:
        fh.write(text)

    print(f'[yaml] yolo_model -> {model}')
    print(f'[yaml] yolo_conf_threshold -> {args.conf}')
    print(f'[yaml] updated {args.yaml}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
