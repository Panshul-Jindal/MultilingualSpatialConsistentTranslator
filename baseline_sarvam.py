import argparse
import json
import os
import sys
import time

from sarvamai import SarvamAI
from sarvamai.speech_to_text_job.job import SpeechToTextJob


def run_sarvam_baseline(
    audio_path: str = "data/Sequentially.wav",
    num_speakers: int = 3,
    output_dir: str = "data/output",
    mode: str = "translate",
) -> dict:
    """Run Sarvam AI Batch STT with Diarization as a baseline."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("Set SARVAM_API_KEY environment variable first.")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    client = SarvamAI(api_subscription_key=api_key)
    filename = os.path.basename(audio_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(" " * 22 + "STAGE 1: SARVAM BASELINE DIARIZATION")
    print("=" * 80)
    print(f"Audio File   : {audio_path}")
    print(f"Num Speakers : {num_speakers}")
    print(f"Model        : saaras:v4 (mode='{mode}' -> English output)")

    start_time = time.time()

    # 1. Initialize Batch STT Job with num_speakers = 3 and mode = translate
    init_res = client.speech_to_text_job.initialise(
        job_parameters={
            "model": "saaras:v4",
            "language_code": "unknown",
            "with_diarization": True,
            "num_speakers": num_speakers,
            "with_timestamps": True,
            "mode": mode,
        }
    )
    job_id = init_res.job_id
    print(f"\n[1/4] Job initialized. Job ID: {job_id}")

    # 2. Upload audio
    print(f"[2/4] Uploading '{filename}'...")
    job = SpeechToTextJob(job_id=job_id, client=client.speech_to_text_job)
    job.upload_files([audio_path])

    # 3. Start job and poll for completion
    print("[3/4] Processing started. Polling server...", flush=True)
    client.speech_to_text_job.start(job.job_id)

    poll_interval = 3
    timeout = 600
    poll_start = time.monotonic()

    while True:
        status = job.get_status()
        state = status.job_state
        elapsed_sec = int(time.time() - start_time)
        print(f"      [+{elapsed_sec:2d}s] Job state: {state}", flush=True)

        if state.lower() in {"completed", "failed"}:
            break

        if time.monotonic() - poll_start > timeout:
            raise TimeoutError(f"Job {job.job_id} timed out after {timeout}s.")

        time.sleep(poll_interval)

    elapsed = time.time() - start_time

    if status.job_state.lower() != "completed":
        raise RuntimeError(f"Job {job_id} failed with state: {status.job_state}")

    print(f"      Job finished in {elapsed:.2f}s with status: {status.job_state}", flush=True)

    # 4. Download and parse results
    print("[4/4] Downloading output...")
    job.download_outputs(output_dir=output_dir)
    result_file = os.path.join(output_dir, f"{filename}.json")

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["elapsed_seconds"] = round(elapsed, 2)

    # Display results
    print("\n" + "=" * 80)
    print(" " * 28 + "BASELINE RESULTS")
    print("=" * 80)
    print(f"Detected Language : {data.get('language_code')} (probability: {data.get('language_probability')})")
    print(f"Full Transcript   : {data.get('transcript')}\n")

    entries = data.get("diarized_transcript", {}).get("entries", [])
    print(f"Diarized Segments ({len(entries)} detected):")
    if not entries:
        print("  [Warning] No diarized entries returned. All speech placed in full transcript.")
    else:
        for idx, entry in enumerate(entries, start=1):
            spk = entry.get("speaker_id")
            s = entry.get("start_time_seconds")
            e = entry.get("end_time_seconds")
            txt = entry.get("transcript")
            print(f"  #{idx:02d} [{s:05.2f}s - {e:05.2f}s] Speaker {spk}: \"{txt}\"")

    # Save clean baseline summary
    summary_path = os.path.join(output_dir, "sarvam_baseline_diarization.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFull baseline output saved to: {summary_path}")
    print("=" * 80)
    return data


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Sarvam Batch Diarization Baseline")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="data/Sequentially.wav",
        help="Audio file path (default: data/Sequentially.wav)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=3,
        help="Number of speakers (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/output",
        help="Output directory (default: data/output)",
    )
    parser.add_argument(
        "--mode",
        default="translate",
        choices=["translate", "transcribe"],
        help="Mode: 'translate' for English, 'transcribe' for Telugu (default: translate)",
    )

    args = parser.parse_args()
    run_sarvam_baseline(
        audio_path=args.audio_path,
        num_speakers=args.num_speakers,
        output_dir=args.output_dir,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
