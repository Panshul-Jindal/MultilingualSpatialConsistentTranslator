import argparse
import json
import os
import shutil
import time
from typing import List

from diarizer import PyAnnoteDiarizer, SpeakerTurn
from translator import SarvamTranslator, TranslatedSpeakerSegment


def normalize_audio(input_wav: str, output_wav: str) -> str:
    """Normalize audio loudness using ffmpeg loudnorm into a separate output file."""
    import subprocess
    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000",
        output_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_wav


def process_audio(
    audio_path: str,
    output_dir: str = "data/output",
    num_speakers: int | None = 3,
    keep_chunks: bool = False,
    normalize: bool = True,
) -> List[TranslatedSpeakerSegment]:
    """Orchestrate PyAnnote Diarization + Sarvam AI 2-Stage Translation."""
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    chunks_dir = os.path.join(output_dir, "temp_chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    print("=" * 80)
    print(" " * 18 + "MULTILINGUAL SPATIAL CONSISTENT TRANSLATOR")
    print("=" * 80)
    print(f"Input Audio  : {audio_path}")
    print(f"Num Speakers : {num_speakers if num_speakers is not None else 'Auto-detect'}")
    print(f"Output Dir   : {output_dir}")

    # Optional Preprocessing: Loudness normalization to a separate file
    active_audio_path = audio_path
    if normalize:
        norm_file = os.path.join(output_dir, "normalized_input.wav")
        print(f"\n>>> PREPROCESSING: Loudness Normalization <<<")
        print(f"Normalizing audio dynamics to separate file: {norm_file}")
        print("(Original input file remains completely unchanged)")
        active_audio_path = normalize_audio(audio_path, norm_file)

    # --- STAGE 1: Speaker Diarization with PyAnnote 3.1 ---
    print("\n>>> STAGE 1: Speaker Diarization (PyAnnote 3.1) <<<")
    diarizer = PyAnnoteDiarizer()
    t0 = time.time()
    raw_turns = diarizer.diarize(active_audio_path, num_speakers=num_speakers)
    diarization_time = time.time() - t0

    # Log raw segments before any merging
    unique_speakers = set(t.speaker_id for t in raw_turns)
    print(f"\nRaw diarization: {len(raw_turns)} segment(s), {len(unique_speakers)} unique speaker(s) detected.")
    for i, t in enumerate(raw_turns, 1):
        print(f"  Raw #{i}: [{t.start:.2f}s - {t.end:.2f}s] {t.speaker_id} ({t.duration:.2f}s)")

    # If clustering collapsed all segments into one speaker but we expected more,
    # treat each raw segment as a separate speaker turn (VAD-based fallback).
    if len(unique_speakers) == 1 and num_speakers and num_speakers > 1 and len(raw_turns) >= num_speakers:
        print(f"\n[Fallback] Clustering found only 1 speaker ID but {len(raw_turns)} voice regions.")
        print(f"           Using raw VAD segments as individual speaker turns.")
        for i, t in enumerate(raw_turns):
            raw_turns[i] = SpeakerTurn(
                speaker_id=f"SPEAKER_{i:02d}",
                start=t.start,
                end=t.end,
            )
        merged_turns = raw_turns
    else:
        merged_turns = diarizer.merge_adjacent_turns(raw_turns, max_gap=0.4)

    print(f"\nDiarization finished in {diarization_time:.2f}s ({len(merged_turns)} speaker turn(s) to process).")

    if not merged_turns:
        print("Warning: No speech turns detected in the audio file.")
        return []

    # --- STAGE 2: 2-Stage Speech-to-Text & Translation with Sarvam AI ---
    print("\n>>> STAGE 2: Multilingual STT & English Translation (Sarvam AI) <<<")
    translator = SarvamTranslator()
    results: List[TranslatedSpeakerSegment] = []
    detected_lang_hint = "unknown"  # Will be updated from first successful long segment

    for i, turn in enumerate(merged_turns, start=1):
        chunk_filename = f"chunk_{i:02d}_{turn.speaker_id}_{turn.start:.2f}s_{turn.end:.2f}s.wav"
        chunk_path = os.path.join(chunks_dir, chunk_filename)

        # Pad segments with extra context so ASR has enough audio.
        # Use more padding for very short segments.
        pad_sec = 0.5 if turn.duration < 1.0 else 0.3
        slice_start = max(0.0, turn.start - pad_sec)
        slice_end = turn.end + pad_sec
        diarizer.slice_wav_segment(active_audio_path, slice_start, slice_end, chunk_path)

        # For short segments (<1s), use the language detected from longer segments
        # instead of "unknown" to avoid misclassification.
        lang_for_segment = detected_lang_hint if turn.duration < 1.0 and detected_lang_hint != "unknown" else "unknown"

        print(f"\nProcessing Turn #{i}: [{turn.start:.2f}s - {turn.end:.2f}s] {turn.speaker_id} ({turn.duration:.2f}s, lang_hint={lang_for_segment})")

        segment_res = translator.process_segment(
            audio_segment_path=chunk_path,
            speaker_id=turn.speaker_id,
            start=turn.start,
            end=turn.end,
            language_code=lang_for_segment,
            target_lang="en-IN",
        )
        results.append(segment_res)

        # Update language hint from longer, more reliable segments
        if turn.duration >= 1.0 and segment_res.source_language not in ("unknown", "en-IN", None):
            detected_lang_hint = segment_res.source_language

        print(f"  • Spoken ({segment_res.source_language}) : {segment_res.source_transcript}")
        print(f"  • English Translation : {segment_res.translated_transcript}")

    # Clean up temporary audio slices if not requested
    if not keep_chunks:
        shutil.rmtree(chunks_dir, ignore_errors=True)

    total_time = time.time() - start_total_time

    # --- Display Summary Table ---
    print("\n" + "=" * 80)
    print(" " * 28 + "FINAL CONVERSATION LOG")
    print("=" * 80)
    for seg in results:
        print(f"[{seg.start:05.2f}s - {seg.end:05.2f}s] {seg.speaker_id} ({seg.source_language}):")
        print(f"   Original: \"{seg.source_transcript}\"")
        print(f"   English : \"{seg.translated_transcript}\"\n")

    print(f"Total Pipeline Runtime: {total_time:.2f}s (Diarization: {diarization_time:.2f}s)")

    # Save to JSON
    json_path = os.path.join(output_dir, "diarized_translation.json")
    output_payload = {
        "audio_file": audio_path,
        "total_runtime_seconds": round(total_time, 2),
        "diarization_runtime_seconds": round(diarization_time, 2),
        "segments": [seg.__dict__ for seg in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved complete results to: {json_path}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PyAnnote 3.1 Diarization + Sarvam AI 2-Stage Multilingual Translator"
    )
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="data/Sequentially.wav",
        help="Path to WAV audio file (default: data/Sequentially.wav)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=3,
        help="Expected number of speakers (default: 3, set 0 for auto-detection)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Directory to save output JSON and chunks (default: data/output)",
    )
    parser.add_argument(
        "--keep-chunks",
        action="store_true",
        help="Keep sliced audio chunks in output folder for debugging",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable automatic loudness normalization",
    )
    parser.set_defaults(normalize=True)

    args = parser.parse_args()
    num_spk = args.num_speakers if args.num_speakers > 0 else None

    process_audio(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        num_speakers=num_spk,
        keep_chunks=args.keep_chunks,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
