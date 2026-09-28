#!/usr/bin/env python3
"""
demo.py
-------
End-to-end demonstration of the full pipeline using SYNTHETIC room
data, so it runs anywhere without a microphone or real recordings.

It simulates two different rooms (a small, "dry" room and a large,
"live" hall), runs the full analysis pipeline on each, saves plots
and fingerprints, and finally compares the two rooms.

For real use: replace `synthetic.simulate_recording(...)` with
`io_utils.load_wav("your_clap.wav")` or `io_utils.try_record(...)`.
"""

import os
import json

from src import synthetic
from src import io_utils
from src import pipeline as pipeline_mod
from src import fingerprint as fp_mod
from src import compare as compare_mod

FS = 44100
OUT_DIR = "output"


def analyze_room(room_name, rt60_target, n_reflections, out_subdir, seed):
    print(f"\n--- Simulating and analyzing: {room_name} (target RT60 ~ {rt60_target}s) ---")

    rir = synthetic.generate_synthetic_rir(
        fs=FS, rt60=rt60_target, duration=max(1.5, rt60_target * 2),
        n_early_reflections=n_reflections, seed=seed,
    )
    recorded = synthetic.simulate_recording(rir, FS, snr_db=28.0, seed=seed)

    out_dir = os.path.join(OUT_DIR, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    io_utils.save_wav(os.path.join(out_dir, "simulated_clap_recording.wav"), recorded, FS)

    analyzer = pipeline_mod.RoomAcousticAnalyzer(room_name=room_name)
    analyzer.run(recorded, FS, is_clap=True)

    plot_path = analyzer.plot_summary(out_dir)
    fp_path = os.path.join(out_dir, "fingerprint.json")
    fp_mod.save_fingerprint(analyzer.results["fingerprint"], fp_path)
    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(analyzer.report_text())

    print(analyzer.report_text())
    print(f"Saved plot: {plot_path}")
    print(f"Saved fingerprint: {fp_path}")

    return analyzer.results["fingerprint"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fp_small_room = analyze_room(
        room_name="Small Dry Room", rt60_target=0.3, n_reflections=8,
        out_subdir="small_room", seed=1,
    )
    fp_hall = analyze_room(
        room_name="Large Reverberant Hall", rt60_target=1.8, n_reflections=20,
        out_subdir="hall", seed=2,
    )
    fp_office = analyze_room(
        room_name="Office (similar to small room)", rt60_target=0.35, n_reflections=9,
        out_subdir="office", seed=3,
    )

    print("\n--- Room-to-Room Comparisons ---")
    comparisons = {
        "small_room_vs_hall": compare_mod.compare_fingerprints(fp_small_room, fp_hall),
        "small_room_vs_office": compare_mod.compare_fingerprints(fp_small_room, fp_office),
        "hall_vs_office": compare_mod.compare_fingerprints(fp_hall, fp_office),
    }

    for name, result in comparisons.items():
        print(f"\n{name}:")
        print(json.dumps(result, indent=2))

    with open(os.path.join(OUT_DIR, "comparisons.json"), "w") as f:
        json.dump(comparisons, f, indent=2)

    print(f"\nAll outputs saved under: {os.path.abspath(OUT_DIR)}/")
    print("(Small room and Office should show HIGH similarity;")
    print(" both should show LOW similarity to the Hall.)")


if __name__ == "__main__":
    main()
