# Room Acoustic Fingerprint & Analysis System

cd room_acoustic_fingerprint
python -m pip install -r requirements.txt
python -m streamlit run app.py


A Python DSP project (CSE 200) that analyzes the acoustic character of a
room from a recorded impulse sound (a hand clap), estimates the **Room
Impulse Response (RIR)**, and derives a compact **acoustic fingerprint**
that can be compared across rooms.

## Pipeline

```
Record Audio / Clap
      │
      ▼
Sampling & Pre-processing  (resample, DC removal, normalize)
      │
      ▼
Noise Filtering  (band-pass filter, spectral noise gate)
      │
      ▼
Correlation Analysis  (autocorrelation / echo lag structure)
      │
      ▼
Echo & Reflection Detection  (peak picking on the envelope)
      │
      ▼
Room Impulse Response  h[n]
      │
      ├──► FFT / Frequency Response  H(f)
      │
      ├──► STFT / Spectrogram
      │
      └──► Energy Decay Analysis (Schroeder integration)
                  │
                  ▼
             RT60 Estimation (EDT / T20 / T30)
                  │
                  ▼
          Acoustic Feature Extraction
                  │
                  ▼
          Room Acoustic Fingerprint
                  │
                  ▼
          Room-to-Room Comparison
```

## Where each theoretical concept lives in the code

| Concept                              | File                       | Notes |
|---------------------------------------|----------------------------|-------|
| Sampling & discrete-time signals      | `src/io_utils.py`, `src/preprocessing.py` | Loading/resampling turns a continuous pressure wave into x[n] |
| LTI systems                           | `src/preprocessing.py`, `src/rir.py` | Filters and the room itself are modeled as LTI systems |
| Convolution                           | `src/synthetic.py` (`simulate_recording`), `src/preprocessing.py` (filtering) | y[n] = x[n] * h[n] |
| Correlation                           | `src/correlation.py` | Cross-/auto-correlation for lag/echo detection |
| FFT                                    | `src/spectral.py` | `np.fft.rfft` on h[n] |
| Frequency response                     | `src/spectral.py` | H(f) = FFT{h[n]} |
| STFT & spectrogram                     | `src/stft.py` | `scipy.signal.stft` |
| Energy decay analysis                  | `src/energy_decay.py` | Schroeder backward integration |
| RT60 estimation                        | `src/rt60.py` | Linear regression on the EDC (EDT/T20/T30), plus C50/C80/DRR |
| Peak / reflection detection            | `src/peak_detection.py` | Envelope + `scipy.signal.find_peaks` |
| Acoustic feature extraction            | `src/fingerprint.py` | Combines all of the above into named features |
| Room acoustic fingerprint              | `src/fingerprint.py` | Feature dict + fixed-length numeric vector |
| Room-to-room comparison                | `src/compare.py` | Cosine similarity, Euclidean distance, per-feature diffs |

`src/pipeline.py` is the orchestrator (`RoomAcousticAnalyzer`) that runs
every stage in order and produces plots + a text report.

## Project structure

```
room_acoustic_fingerprint/
├── README.md
├── requirements.txt
├── main.py            # CLI: analyze a .wav / record live, and compare fingerprints
├── demo.py            # Runs the whole pipeline on synthetic rooms (no mic needed)
├── src/
│   ├── io_utils.py         # load/save .wav, optional live mic recording
│   ├── preprocessing.py    # resample, DC removal, band-pass filter, noise gate
│   ├── correlation.py      # cross-/auto-correlation
│   ├── rir.py               # room impulse response estimation (clap or deconvolution)
│   ├── peak_detection.py   # reflection/echo peak picking
│   ├── spectral.py          # FFT, frequency response, octave-band energy
│   ├── stft.py               # STFT / spectrogram
│   ├── energy_decay.py      # Schroeder integration (energy decay curve)
│   ├── rt60.py                # RT60 (EDT/T20/T30), clarity (C50/C80), DRR
│   ├── fingerprint.py        # feature extraction + fingerprint vector
│   ├── compare.py             # similarity/distance metrics between fingerprints
│   ├── synthetic.py          # synthetic RIR + clap generator (for testing/demo)
│   └── pipeline.py            # RoomAcousticAnalyzer: orchestrates everything + plots
└── output/                    # created by demo.py / main.py (plots, fingerprints, reports)
```

## 🖥️ Web UI (recommended — everything in one screen)

The whole project is also available as a single-page web dashboard
(`app.py`, built with Streamlit). It covers every stage: recording/
upload, running the full pipeline, viewing every plot, and comparing
rooms — no terminal commands needed after launch.

```bash
pip install -r requirements.txt
streamlit run app.py
```

This opens a browser tab (usually `http://localhost:8501`) with three tabs:

1. **🎙️ Analyze a Room** — upload a `.wav`, record straight from your
   browser mic, or generate a synthetic demo room. Click **Run Full
   Analysis** to execute the entire pipeline and see the RT60/clarity/DRR
   metrics plus all six plots (impulse response, frequency response,
   spectrogram, energy decay curve, autocorrelation, RT60-per-band).
2. **📚 Room Library** — every room you've analyzed, in one comparison
   table, with CSV export.
3. **⚖️ Compare Rooms** — pick any two analyzed rooms to get a similarity
   score and a breakdown of the biggest acoustic differences.

Rooms you analyze are automatically saved under `output/<room_name>/`,
so they're still in your Room Library next time you launch the app.

The CLI (`main.py`) and `demo.py` from the sections below still work
exactly the same and share the same underlying `src/` pipeline — use
whichever fits (UI for exploring interactively, CLI/demo for scripting
or batch runs).

## Installation

```bash
pip install -r requirements.txt
```

`sounddevice` and `soundfile` are optional — only needed for live
microphone recording / extra audio formats. Everything else works with
just `numpy`, `scipy`, and `matplotlib`.

## Quick start — run the demo (no microphone needed)

```bash
python demo.py
```

This simulates three rooms (a small dry room, a large reverberant hall,
and an office similar to the small room), runs the **entire** pipeline
on each, and saves for each room under `output/<room>/`:
- `analysis_summary.png` — 6-panel figure: impulse response with
  detected reflections, frequency response, spectrogram, energy decay
  curve with RT60 fit, autocorrelation, and RT60 per octave band.
- `fingerprint.json` — the room's acoustic fingerprint (features + vector).
- `report.txt` — a human-readable summary (RT60, clarity, DRR, etc).
- `simulated_clap_recording.wav` — the synthetic recording that was analyzed.

It finishes by comparing all three rooms pairwise
(`output/comparisons.json`) — expect the small room and office to be
highly similar, and both to be dissimilar to the hall.

## Using it with a real recording

1. Clap once, sharply, in the room you want to analyze and record it
   (phone voice memo exported to `.wav`, Audacity, etc.), **or** run
   with a laptop microphone via `--record`.
2. Analyze it:

```bash
# From a .wav file
python main.py analyze --input my_room_clap.wav --name "Bedroom" --output out/bedroom

# Or record live for 3 seconds (needs `sounddevice` + a working mic)
python main.py analyze --record --duration 3 --name "Bedroom" --output out/bedroom
```

3. Compare two rooms you've already analyzed:

```bash
python main.py compare --fp1 out/bedroom/fingerprint.json --fp2 out/office/fingerprint.json
```

### Recording tips for good results
- Clap once, sharply (a flat-palm clap gives a broadband, impulsive excitation).
- Minimize background noise/talking during and just after the clap.
- Leave ~1–2 seconds of "tail" recording after the clap so the decay can be captured.
- Keep the mic a consistent distance from the clap across rooms you want to compare.

## Known limitations / possible extensions
- The clap-as-impulse method is a classroom-friendly approximation; a
  logarithmic sine-sweep excitation + `rir.estimate_rir_deconvolution()`
  gives a cleaner, higher-SNR impulse response for more rigorous work.
- RT60 estimates on very short/noisy recordings can be `None` (not enough
  decay range to fit) — this is expected and reported as `n/a`.
- Octave-band filtering near the edges of the audible range (e.g. the
  125 Hz band) can be less stable — feel free to swap `octave_band_energy`
  in `src/spectral.py` for a longer/higher-order filter if that matters
  for your report.
- The fingerprint vector is a simple, interpretable hand-built feature
  vector; a natural extension is dimensionality reduction (PCA) or a
  learned embedding for larger fingerprint databases.
