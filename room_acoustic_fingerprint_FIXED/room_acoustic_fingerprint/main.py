#!/usr/bin/env python3
"""
main.py
-------
Command-line interface for the Room Acoustic Fingerprint & Analysis System.

Usage:
    # Analyze a recorded clap (.wav) and save a fingerprint + plots
    python main.py analyze --input clap_room1.wav --name "Living Room" --output out/room1

    # Record live from the microphone (if available) instead of a file
    python main.py analyze --record --duration 3 --name "Office" --output out/office

    # Compare two previously saved fingerprints
    python main.py compare --fp1 out/room1/fingerprint.json --fp2 out/office/fingerprint.json
"""

import argparse
import json
import os
import sys

from src import io_utils
from src import pipeline as pipeline_mod
from src import fingerprint as fp_mod
from src import compare as compare_mod


def cmd_analyze(args):
    if args.record:
        signal, fs = io_utils.try_record(duration_sec=args.duration)
    elif args.input:
        signal, fs = io_utils.load_wav(args.input)
    else:
        print("Error: supply --input <file.wav> or --record", file=sys.stderr)
        sys.exit(1)

    analyzer = pipeline_mod.RoomAcousticAnalyzer(room_name=args.name)
    analyzer.run(signal, fs)

    os.makedirs(args.output, exist_ok=True)
    plot_path = analyzer.plot_summary(args.output)
    fp_path = os.path.join(args.output, "fingerprint.json")
    fp_mod.save_fingerprint(analyzer.results["fingerprint"], fp_path)

    report_path = os.path.join(args.output, "report.txt")
    with open(report_path, "w") as f:
        f.write(analyzer.report_text())

    print(analyzer.report_text())
    print(f"\nSaved: {plot_path}\nSaved: {fp_path}\nSaved: {report_path}")


def cmd_compare(args):
    fp1 = fp_mod.load_fingerprint(args.fp1)
    fp2 = fp_mod.load_fingerprint(args.fp2)
    result = compare_mod.compare_fingerprints(fp1, fp2)
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved comparison to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Room Acoustic Fingerprint & Analysis System")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyze = sub.add_parser("analyze", help="Analyze a room recording and build its fingerprint")
    p_analyze.add_argument("--input", type=str, help="Path to a .wav recording of a clap")
    p_analyze.add_argument("--record", action="store_true", help="Record live from microphone instead")
    p_analyze.add_argument("--duration", type=float, default=3.0, help="Recording duration in seconds")
    p_analyze.add_argument("--name", type=str, default="room", help="Name/label for this room")
    p_analyze.add_argument("--output", type=str, default="output/room", help="Output directory")
    p_analyze.set_defaults(func=cmd_analyze)

    p_compare = sub.add_parser("compare", help="Compare two saved fingerprints")
    p_compare.add_argument("--fp1", type=str, required=True, help="Path to first fingerprint.json")
    p_compare.add_argument("--fp2", type=str, required=True, help="Path to second fingerprint.json")
    p_compare.add_argument("--output", type=str, default=None, help="Optional path to save comparison JSON")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
