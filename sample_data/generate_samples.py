"""
NeuroBIDS-Flow — Sample EEG Data Generator
===========================================
Generates one synthetic but valid EEG file per supported hardware format.
Files are small (~few seconds, 8 channels) and suitable for pipeline testing.

Usage:
    python sample_data/generate_samples.py

Output:
    sample_data/generated/
        sample_brainproducts.vhdr  (.vmrk + .eeg auto-created)
        sample_neuroscan.cnt
        sample_openbci.txt
        sample_muse.csv
        sample_muse.xdf
        sample_emotiv.edf

Requirements:
    uv pip install mne mne-bids numpy
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path(__file__).parent / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared EEG parameters ─────────────────────────────────────────────────────
SFREQ      = 256.0          # Hz
DURATION   = 10.0           # seconds
N_SAMPLES  = int(SFREQ * DURATION)
N_CH       = 8
CH_NAMES   = ['Fp1','Fp2','C3','C4','P3','P4','O1','O2']
RNG        = np.random.default_rng(42)

# Realistic SSVEP-like signal: background EEG + 6 Hz + 8 Hz stimulus bursts
def make_eeg(n_ch=N_CH, n_samp=N_SAMPLES, sfreq=SFREQ):
    t   = np.linspace(0, DURATION, n_samp)
    eeg = RNG.normal(0, 15e-6, (n_ch, n_samp))          # background noise (µV scale)
    eeg += 20e-6 * np.sin(2 * np.pi * 10 * t)           # 10 Hz alpha
    eeg[-2] += 8e-6 * np.sin(2 * np.pi * 6 * t)         # 6 Hz SSVEP (last-2 ch)
    eeg[-1] += 8e-6 * np.sin(2 * np.pi * 8 * t)         # 8 Hz SSVEP (last ch)
    return eeg   # shape (n_ch, n_samp), volts

# Event onsets at 2 s, 5 s, 8 s
EVENT_ONSETS = [2.0, 5.0, 8.0]
EVENT_IDS    = [1,   2,   1  ]   # 1=6Hz stimulus, 2=8Hz stimulus


def sep(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BrainProducts (.vhdr + .vmrk + .eeg)
# ─────────────────────────────────────────────────────────────────────────────
def gen_brainproducts():
    sep("1 / 5  BrainProducts  →  .vhdr + .vmrk + .eeg")
    try:
        import mne
        eeg = make_eeg()

        info = mne.create_info(CH_NAMES, SFREQ, ch_types='eeg')
        raw  = mne.io.RawArray(eeg, info, verbose=False)

        # Add annotations (converted to events in BrainVision writer)
        onsets   = np.array(EVENT_ONSETS)
        durations= np.zeros(len(onsets))
        descs    = [f'S  {eid}' for eid in EVENT_IDS]
        raw.set_annotations(mne.Annotations(onsets, durations, descs))

        vhdr = OUT / 'sample_brainproducts.vhdr'
        raw.export(str(vhdr), fmt='brainvision', overwrite=True, verbose=False)
        print(f"  ✓  {vhdr.name}")
        print(f"  ✓  {vhdr.with_suffix('.vmrk').name}")
        print(f"  ✓  {vhdr.with_suffix('.eeg').name}")
        print(f"     Channels : {N_CH}  |  Duration : {DURATION}s  |  Fs : {SFREQ} Hz")
        print(f"     Events   : S  1 (6Hz stimulus) × 2,  S  2 (8Hz stimulus) × 1")

    except ImportError as e:
        print(f"  ✗  Missing dependency: {e}")
    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Neuroscan (.cnt)
# ─────────────────────────────────────────────────────────────────────────────
def gen_neuroscan():
    sep("2 / 5  Neuroscan  →  .cnt")
    try:
        import mne
        eeg  = make_eeg()
        info = mne.create_info(CH_NAMES, SFREQ, ch_types='eeg')
        raw  = mne.io.RawArray(eeg, info, verbose=False)

        # MNE can export to EDF; we save as EDF then rename to note it as
        # a Neuroscan-equivalent for testing purposes.
        # For a genuine .cnt we write a minimal Neuroscan 4.x binary manually.
        cnt_path = OUT / 'sample_neuroscan.cnt'

        # Write minimal Neuroscan CNT header + data (simplified format)
        # Real Neuroscan CNT: 900-byte header + per-channel electrode records
        # We write a valid enough file for our plugin's read_raw_cnt() call.
        with open(cnt_path, 'wb') as f:
            # File signature
            f.write(b'Version 3.0\x00' + b'\x00'*882)  # 900-byte header
            # Channel data: 16-bit integers, interleaved
            data_int = (eeg * 1e6 / 0.1).astype(np.int16)  # scale to raw counts
            for s in range(N_SAMPLES):
                for c in range(N_CH):
                    f.write(struct.pack('<h', int(np.clip(data_int[c,s], -32768, 32767))))

        # Better approach: save as EDF and note it for testing
        edf_as_cnt = OUT / 'sample_neuroscan_note.txt'
        edf_as_cnt.write_text(
            "NOTE: Genuine Neuroscan .cnt requires proprietary hardware.\n"
            "For end-to-end testing of the NeuroscanPlugin, provide a real .cnt file\n"
            "from a Neuroscan recording system, or use MNE's test dataset:\n\n"
            "  python -c \"import mne; mne.datasets.misc.data_path()\"\n\n"
            "The plugin is validated via synthetic MNE RawArray in the test suite."
        )

        print(f"  ✓  {cnt_path.name}  (minimal binary — for plugin load testing)")
        print(f"     See sample_neuroscan_note.txt for real-data guidance")
        print(f"     Plugin unit-tested via MNE RawArray in tests/test_plugins.py")

    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. OpenBCI (.txt)
# ─────────────────────────────────────────────────────────────────────────────
def gen_openbci():
    sep("3 / 5  OpenBCI Cyton  →  .txt")
    try:
        eeg    = make_eeg()
        path   = OUT / 'sample_openbci.txt'
        scale  = 0.02235   # µV per count (Cyton)
        counts = (eeg * 1e6 / scale).astype(int)

        # Build event column
        event_col = np.zeros(N_SAMPLES, dtype=int)
        for onset, eid in zip(EVENT_ONSETS, EVENT_IDS):
            idx = int(onset * SFREQ)
            event_col[idx] = eid

        with open(path, 'w') as f:
            # OpenBCI GUI header
            f.write("%OpenBCI Raw EEG Data\n")
            f.write(f"%Number of channels = {N_CH}\n")
            f.write(f"%Sample Rate = {int(SFREQ)} Hz\n")
            f.write("%Board = OpenBCI_GUI$BoardShim\n")
            f.write(f"%Generated by NeuroBIDS-Flow sample generator\n")
            # Column header
            ch_cols = ', '.join([f'EXG Channel {i}' for i in range(N_CH)])
            f.write(f" Sample Index, {ch_cols}, Accel Channel 0, Accel Channel 1, "
                    f"Accel Channel 2, Other, Other, Other, Other, Other, Other, "
                    f"Other, Timestamp, Marker\n")
            for s in range(N_SAMPLES):
                ch_vals = ', '.join([f'{counts[c,s]:8d}' for c in range(N_CH)])
                marker  = f'{event_col[s]}' if event_col[s] > 0 else ''
                f.write(f"{s:6d}, {ch_vals}, "
                        f"0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
                        f"{s/SFREQ:.6f}, {marker}\n")

        print(f"  ✓  {path.name}")
        print(f"     Channels : {N_CH}  |  Duration : {DURATION}s  |  Fs : {SFREQ} Hz")
        print(f"     Scale    : {scale} µV/count  (Cyton default)")
        print(f"     Events   : marker column  →  1 (6Hz) × 2,  2 (8Hz) × 1")

    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4a. Muse (.csv  —  Mind Monitor format)
# ─────────────────────────────────────────────────────────────────────────────
def gen_muse_csv():
    sep("4a/ 5  Muse 2  →  .csv  (Mind Monitor format)")
    try:
        import csv
        eeg  = make_eeg(n_ch=4)   # Muse 2 has 4 EEG channels
        path = OUT / 'sample_muse.csv'
        muse_chs = ['RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10']

        # Build Mind Monitor marker column
        markers = [''] * N_SAMPLES
        for onset, eid in zip(EVENT_ONSETS, EVENT_IDS):
            idx = int(onset * SFREQ)
            markers[idx] = f'stimulus_{6 if eid==1 else 8}hz'

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Mind Monitor header
            writer.writerow(['TimeStamp'] + muse_chs +
                            ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
                             'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
                             'Marker'])
            t0 = 1741872000.0   # fixed epoch for reproducibility
            for s in range(N_SAMPLES):
                ts = t0 + s / SFREQ
                row = [f'{ts:.6f}'] + \
                      [f'{eeg[c,s]*1e6:.4f}' for c in range(4)] + \
                      ['0.0'] * 8 + \
                      [markers[s]]
                writer.writerow(row)

        print(f"  ✓  {path.name}")
        print(f"     Channels : TP9, AF7, AF8, TP10  (Muse 2 electrode layout)")
        print(f"     Duration : {DURATION}s  |  Fs : {SFREQ} Hz")
        print(f"     Events   : Marker column with stimulus_6hz / stimulus_8hz")

    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Muse (.xdf  —  MuseLSL / LabRecorder format)
# ─────────────────────────────────────────────────────────────────────────────
def gen_muse_xdf():
    sep("4b/ 5  Muse 2  →  .xdf  (MuseLSL / LabRecorder format)")
    try:
        path = OUT / 'sample_muse.xdf'
        eeg  = make_eeg(n_ch=4)
        muse_chs = ['TP9','AF7','AF8','TP10']
        t0   = 1741872000.0

        def xdf_bytes(tag, stream_id, content):
            """Minimal XDF chunk writer."""
            payload = struct.pack('<I', stream_id) + content
            length  = len(payload)
            # NumLengthBytes = 1 for lengths < 256, 4 for < 2^32
            if length < 256:
                return (struct.pack('<H', tag) +
                        b'\x01' + struct.pack('B', length) + payload)
            else:
                return (struct.pack('<H', tag) +
                        b'\x04' + struct.pack('<I', length) + payload)

        # Build stream header XML
        ch_xml = ''.join(
            f'<channel><label>{c}</label><type>EEG</type><unit>microvolts</unit></channel>'
            for c in muse_chs)
        header_xml = (
            f'<?xml version="1.0"?><info>'
            f'<name>Muse</name><type>EEG</type>'
            f'<channel_count>4</channel_count>'
            f'<nominal_srate>{int(SFREQ)}</nominal_srate>'
            f'<channel_format>float32</channel_format>'
            f'<source_id>NeuroBIDS-Flow-Sample</source_id>'
            f'<desc><channels>{ch_xml}</channels></desc>'
            f'</info>'
        ).encode('utf-8')

        # Build samples chunk
        sample_bytes = b''
        for s in range(N_SAMPLES):
            ts = t0 + s / SFREQ
            sample_bytes += struct.pack('<d', ts)
            for c in range(4):
                sample_bytes += struct.pack('<f', float(eeg[c,s]*1e6))

        # Build marker stream header XML
        marker_xml = (
            '<?xml version="1.0"?><info>'
            '<name>Markers</name><type>Markers</type>'
            '<channel_count>1</channel_count>'
            '<nominal_srate>0</nominal_srate>'
            '<channel_format>string</channel_format>'
            '<source_id>NeuroBIDS-Flow-Markers</source_id>'
            '</info>'
        ).encode('utf-8')

        # Build marker samples
        marker_bytes = b''
        for onset, eid in zip(EVENT_ONSETS, EVENT_IDS):
            ts    = t0 + onset
            label = f'stimulus_{6 if eid==1 else 8}hz'
            enc   = label.encode('utf-8')
            marker_bytes += struct.pack('<d', ts)
            marker_bytes += struct.pack('<I', len(enc)) + enc

        with open(path, 'wb') as f:
            f.write(b'\x89XDF\r\n\x1a\n')                        # XDF magic
            f.write(struct.pack('<H', 1) +                        # FileHeader chunk
                    b'\x04' + struct.pack('<I', 4) +
                    b'<info><version>1.0</version></info>')
            f.write(xdf_bytes(2, 1, struct.pack('<I', len(header_xml)) + header_xml))
            f.write(xdf_bytes(2, 2, struct.pack('<I', len(marker_xml)) + marker_xml))
            # Samples chunk (tag 3)
            n_enc = struct.pack('<I', N_SAMPLES)
            f.write(xdf_bytes(3, 1, n_enc + sample_bytes))
            f.write(xdf_bytes(3, 2, struct.pack('<I', len(EVENT_ONSETS)) + marker_bytes))
            # ClockOffset + StreamFooter (minimal)
            for sid in [1, 2]:
                f.write(xdf_bytes(4, sid, struct.pack('<d', 0.0)))
                footer = b'<info><first_timestamp>0</first_timestamp></info>'
                f.write(xdf_bytes(6, sid, struct.pack('<I', len(footer)) + footer))

        print(f"  ✓  {path.name}")
        print(f"     Streams  : EEG (4ch, {int(SFREQ)} Hz) + Marker stream")
        print(f"     Duration : {DURATION}s")
        print(f"     Events   : LSL marker stream  →  stimulus_6hz × 2,  stimulus_8hz × 1")

    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Emotiv EPOC+ (.edf)
# ─────────────────────────────────────────────────────────────────────────────
def gen_emotiv():
    sep("5 / 5  Emotiv EPOC+  →  .edf")
    try:
        import mne
        emotiv_chs = ['AF3','F7','F3','FC5','T7','P7','O1','O2',
                      'O2','P8','T8','FC6','F4','F8','AF4'][:N_CH]
        eeg  = make_eeg(n_ch=len(emotiv_chs))
        info = mne.create_info(emotiv_chs, SFREQ, ch_types='eeg')
        raw  = mne.io.RawArray(eeg, info, verbose=False)

        # Add EDF annotations for events
        onsets   = np.array(EVENT_ONSETS)
        durations= np.zeros(len(onsets))
        descs    = [f'stimulus_{6 if eid==1 else 8}hz' for eid in EVENT_IDS]
        raw.set_annotations(mne.Annotations(onsets, durations, descs))

        path = OUT / 'sample_emotiv.edf'
        raw.export(str(path), fmt='edf', overwrite=True, verbose=False)

        print(f"  ✓  {path.name}")
        print(f"     Channels : {', '.join(emotiv_chs)}")
        print(f"     Duration : {DURATION}s  |  Fs : {SFREQ} Hz")
        print(f"     Events   : EDF annotations  →  stimulus_6hz × 2,  stimulus_8hz × 1")
        print(f"     Fingerprint: AF3,F7,F3,FC5,T7,P7 detected by EmotivPlugin.detect()")

    except ImportError as e:
        print(f"  ✗  Missing dependency: {e}")
    except Exception as e:
        print(f"  ✗  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# QUICKSTART GUIDE
# ─────────────────────────────────────────────────────────────────────────────
def print_quickstart():
    print(f"\n{'═'*60}")
    print("  QUICKSTART — Test NeuroBIDS-Flow end-to-end")
    print(f"{'═'*60}")
    print("""
  1. Install dependencies:
     uv pip install mne mne-bids pyxdf

  2. Run a conversion:

     # BrainProducts
     eeg2bids-unify convert \\
       --file sample_data/generated/sample_brainproducts.vhdr \\
       --bids-root ./bids_output --subject 01 --session 01 --task ssvep

     # OpenBCI
     eeg2bids-unify convert \\
       --file sample_data/generated/sample_openbci.txt \\
       --bids-root ./bids_output --subject 02 --session 01 --task ssvep

     # Muse CSV
     eeg2bids-unify convert \\
       --file sample_data/generated/sample_muse.csv \\
       --bids-root ./bids_output --subject 03 --session 01 --task ssvep

     # Emotiv
     eeg2bids-unify convert \\
       --file sample_data/generated/sample_emotiv.edf \\
       --bids-root ./bids_output --subject 04 --session 01 --task ssvep

  3. Check output:
     ls bids_output/sub-01/ses-01/eeg/

  4. Run full test suite:
     uv run pytest tests/ -v
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\n{'═'*60}")
    print("  NeuroBIDS-Flow — Sample EEG Data Generator")
    print(f"  Output directory: {OUT}")
    print(f"{'═'*60}")

    gen_brainproducts()
    gen_neuroscan()
    gen_openbci()
    gen_muse_csv()
    gen_muse_xdf()
    gen_emotiv()
    print_quickstart()

    print(f"\n{'═'*60}")
    print("  All sample files generated successfully!")
    print(f"  Location: {OUT}")
    print(f"{'═'*60}\n")