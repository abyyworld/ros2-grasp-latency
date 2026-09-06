#!/usr/bin/env python3
"""Turn a recorded /grasp/latency bag into the timing JSONL of docs/FORMATS.md.

The point of doing it this way round (publish a message, record it with
``ros2 bag record``, convert offline) is that no file I/O and no Python
subscriber sit anywhere near the node under measurement. The recorder is
rosbag2's C++ one; this script never runs while a benchmark is running.

It reads the bag with the pure-Python ``rosbags`` library rather than rclpy, so
it also runs outside the container (and outside ROS entirely), which is how it
is tested. The GraspLatency definition is registered from the .msg file in the
workspace, so the script cannot drift from the message it is decoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

MSGTYPE = 'grasp_msgs/msg/GraspLatency'
TOPIC = '/grasp/latency'

# Key order fixed by docs/FORMATS.md, mirrored by the STAGE_* constants in
# grasp_msgs/msg/GraspLatency.msg and by the C++ and Python nodes.
STAGE_KEYS = (
    'decode', 'deproject', 'transform_crop', 'plane',
    'cluster', 'grasp', 'ik', 'traj',
)


def build_typestore(msg_path: Path):
    store = get_typestore(Stores.ROS2_JAZZY)
    store.register(get_types_from_msg(msg_path.read_text(), MSGTYPE))
    return store


def read_records(bag: Path, msg_path: Path):
    store = build_typestore(msg_path)
    out = []
    with Reader(bag) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            raise SystemExit(f'{bag}: no {TOPIC} connection in bag')
        for conn, _, raw in reader.messages(connections=conns):
            out.append(store.deserialize_cdr(raw, conn.msgtype))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', required=True, type=Path)
    ap.add_argument('--msg', required=True, type=Path,
                    help='path to grasp_msgs/msg/GraspLatency.msg')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--impl', required=True,
                    help='value of the "impl" field, e.g. cpp or py')
    ap.add_argument('--transport', default='ros2',
                    help='value of the "transport" field. harness/analyze.py '
                         'groups on (impl, dataset, transport), so this is what '
                         'keeps an inter-process run, a composed run and a '
                         'composed run with intra-process delivery apart. The '
                         'in-process benchmark writes "inproc".')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--frame-count', required=True, type=int,
                    help='frames in the store, to map a sequence number back '
                         'to a frame index')
    ap.add_argument('--warmup', type=int, default=0,
                    help='leading records to discard; seq is renumbered from 0 '
                         'across what remains, per docs/FORMATS.md')
    ap.add_argument('--limit', type=int, default=0,
                    help='keep at most this many records after warm-up, so an '
                         'over-run driver still yields a fixed sample count')
    args = ap.parse_args(argv)

    records = read_records(args.bag, args.msg)
    records.sort(key=lambda m: m.seq)

    if records:
        expected = records[-1].seq - records[0].seq + 1
        dropped = expected - len(records)
    else:
        dropped = 0

    kept = [m for m in records if m.seq >= args.warmup]
    if args.limit:
        kept = kept[:args.limit]

    with args.out.open('w') as fh:
        for seq, m in enumerate(kept):
            gc = [int(v) for v in m.gc_collections]
            row = {
                'impl': args.impl,
                'transport': args.transport,
                'dataset': args.dataset,
                'frame': int(m.seq) % args.frame_count,
                'seq': seq,
                'stage_ns': {k: int(v) for k, v in zip(STAGE_KEYS, m.stage_ns)},
                'total_ns': int(m.compute_ns),
                'end_to_end_ns': int(m.end_to_end_ns),
                'stamp_ns': int(m.stamp.sec) * 1_000_000_000 + int(m.stamp.nanosec),
                'points': int(m.points),
                'cluster_points': int(m.cluster_points),
                'ik_iterations': int(m.ik_iterations),
                'plane_found': bool(m.plane_found),
                'graspable': bool(m.graspable),
                'converged': bool(m.converged),
            }
            # total_ns is the node's own compute, so it lines up with the
            # in-process benchmark. Everything ROS 2 adds ahead of the callback
            # is the difference between end_to_end_ns and it, but only when
            # the stamp was written by a live publisher; see docs/ROS2.md.
            if any(gc):
                row['gc'] = {'gen0': gc[0], 'gen1': gc[1], 'gen2': gc[2]}
            fh.write(json.dumps(row) + '\n')

    print(f'{args.out}: {len(kept)} records [{args.transport}] '
          f'({len(records)} recorded, {args.warmup} warm-up discarded, '
          f'{dropped} lost in flight)', file=sys.stderr)
    if dropped:
        print('WARNING: sequence gaps mean the recorder or the transport lost '
              'samples; frame indices after the first gap are still correct '
              'because they are derived from seq, but the run is short.',
              file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
