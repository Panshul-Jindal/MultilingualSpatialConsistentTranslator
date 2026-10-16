# Multilingual Spatial-Consistent Translator

A speaker-aware, multilingual audio translation pipeline engineered for Indic languages (featuring Telugu → English). 

The system combines neural speaker diarization via **PyAnnote Audio 3.1** on GPU with **Sarvam AI (Saaras:v4)** for native Indic speech recognition and translation. By decoupling speaker segmentation from translation into a synchronized two-stage pipeline, it preserves temporal alignment, speaker identity, and dual-language transcriptions.

---

## Key Features

- **Speaker Diarization & Tracking**: Leverages `pyannote/speaker-diarization-3.1` with millisecond-precision turn detection and Voice Activity Detection (VAD) fallback for short conversational turns.
- **Sub-2s GPU Diarization**: Runs neural diarization on CUDA GPU in ~1.2 seconds, overcoming the high latency of cloud-based batch diarizers.
- **Two-Stage Native Translation**: First transcribes speech in its native language (`te-IN`), then translates to English (`en-IN`), maintaining acoustic fidelity and eliminating cross-talk hallucinations.
- **Audio Dynamic Preprocessing**: Integrated EBU R128 loudness normalization (`ffmpeg loudnorm`) to prevent quiet speakers from being masked as background noise.
- **Clean JSON Export**: Exports timestamps, speaker IDs, source transcripts, and translated text for downstream spatial audio or subtitle rendering.

---

## Pipeline Architecture

```
[ Input Audio (.wav / .m4a) ]
             │
             ▼
┌────────────────────────────────────────┐
│ Preprocessing: Loudness Normalization  │  (EBU R128 Loudnorm)
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│ Stage 1: Neural Speaker Diarization    │  (PyAnnote Audio 3.1 on CUDA)
│ - Voice Activity Detection (VAD)       │
│ - Speaker Embedding & Turn Extraction  │
└───────────────────┬────────────────────┘
                    │  [Speaker Slices + Timestamps]
                    ▼
┌────────────────────────────────────────┐
│ Stage 2: Sarvam AI Indic Engine        │
│ 1. Native ASR (Saaras:v4)              │  Telugu Transcript
│ 2. Indic Translation                   │  English Translation
└───────────────────┬────────────────────┘
                    │
                    ▼
       [ Final Conversation Log ]
       [ data/output/diarized_translation.json ]
```

---

## Benchmark: Sarvam Batch Diarization vs. PyAnnote 3.1 Pipeline

Tested on multi-speaker audio (`Sequentially.wav`, 7.38s):

| Feature | Sarvam AI Batch Diarizer (Baseline) | Our PyAnnote 3.1 + Sarvam 2-Stage Pipeline |
| :--- | :--- | :--- |
| **Speaker Separation** | ❌ Collapsed all 3 speakers into 1 (`Speaker 0`) | ✅ Correctly isolated all 3 speaker turns |
| **Diarization Latency** | 20.77s (Cloud Queue & Polling) | **1.26s** (Local CUDA GPU) |
| **Short Utterance Handling** | ❌ Dropped brief conversational replies | ✅ Captured short 0.44s reply (`"అన్న."` / *"Brother."*) |
| **Translation Quality** | Telugu only (`mode='transcribe'`) | Fluent English + Native Telugu dual transcript |
| **Spatial Alignment** | No individual speaker cuts | Individual speaker audio slices & timestamps preserved |

---

## Quickstart

### Prerequisites

1. **Python 3.13+** and **[uv](https://docs.astral.sh/uv/)** installed.
2. **Hugging Face Token** with access accepted for:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. **Sarvam AI API Key** from [sarvam.ai](https://www.sarvam.ai/).

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Panshul-Jindal/MultilingualSpatialConsistentTranslator.git
cd MultilingualSpatialConsistentTranslator

# Install dependencies using uv
uv sync

# Set your API credentials
export HF_TOKEN="your_huggingface_token"
export SARVAM_API_KEY="your_sarvam_api_key"
```

---

## Usage

### 1. Run the Primary Pipeline (PyAnnote 3.1 + Sarvam 2-Stage)

Run the full end-to-end diarization and translation:

```bash
uv run python main.py data/Sequentially.wav --keep-chunks
```

CLI Options:
- `--num-speakers <N>`: Expected number of speakers (default: `3`, set `0` for auto-detect).
- `--keep-chunks`: Preserves sliced audio chunks per speaker in `data/output/temp_chunks/`.
- `--no-normalize`: Disables automatic loudness normalization.

### 2. Run the Sarvam Baseline (Optional)

To test Sarvam's integrated batch diarizer:

```bash
uv run python baseline_sarvam.py data/Sequentially.wav --num-speakers 3
```

---

## Example Output

```text
================================================================================
                            FINAL CONVERSATION LOG
================================================================================
[00.77s - 03.83s] SPEAKER_00 (te-IN):
   Original: "శైలు ఎలా ఉన్నావు? జగదీష్ ఎలా ఉన్నావు?"
   English : "How are you, Shailu? Jagadish, how are you?"

[03.93s - 04.37s] SPEAKER_01 (te-IN):
   Original: "అన్న."
   English : "Brother."

[05.16s - 06.07s] SPEAKER_02 (te-IN):
   Original: "నేను కూడా బాగున్నాను."
   English : "I'm doing fine as well."

Total Pipeline Runtime: 18.39s (Diarization: 1.26s)
Saved complete results to: data/output/diarized_translation.json
================================================================================
```

---

## Project Structure

```
.
├── diarizer.py           # PyAnnote 3.1 Diarization & Audio Slicing module
├── translator.py         # Sarvam AI 2-Stage Native STT + Text Translation
├── main.py               # Main pipeline runner and CLI orchestrator
├── baseline_sarvam.py    # Sarvam AI Batch STT baseline comparison script
├── batch_comparison.py   # Comprehensive Batch API comparison runner
├── requirements.txt      # Hugging Face Spaces & deployment requirements
├── pyproject.toml        # uv package configuration
└── data/                 # Sample multi-speaker audio files (.wav & .m4a)
```

---

## License

MIT License.
