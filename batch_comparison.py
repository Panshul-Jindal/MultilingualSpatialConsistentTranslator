import json
import os
import sys
import time
from sarvamai import SarvamAI
from sarvamai.speech_to_text_job.job import SpeechToTextJob


def run_batch_job(client: SarvamAI, audio_path: str, mode: str, output_dir: str, num_speakers: int | None = 2) -> dict:
    """Run a Sarvam AI Batch STT job with diarization enabled."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(audio_path)

    print(f"\n--- Initializing Batch Job [mode='{mode}', num_speakers={num_speakers}] for {filename} ---")
    start_time = time.time()

    job_params = {
        "model": "saaras:v4",
        "language_code": "unknown",
        "with_diarization": True,
        "with_timestamps": True,
        "mode": mode,
    }
    if num_speakers is not None:
        job_params["num_speakers"] = num_speakers

    init_res = client.speech_to_text_job.initialise(job_parameters=job_params)
    job_id = init_res.job_id
    print(f"Job ID: {job_id}")

    job = SpeechToTextJob(job_id=job_id, client=client.speech_to_text_job)

    print(f"Uploading '{audio_path}'...")
    job.upload_files([audio_path])

    print("Starting job processing...")
    client.speech_to_text_job.start(job.job_id)

    print("Waiting for job to complete (polling every 5 seconds)...")
    status = job.wait_until_complete(poll_interval=5, timeout=600)
    elapsed = time.time() - start_time
    print(f"Job finished in {elapsed:.2f}s with status: {status.job_state}")

    if status.job_state.lower() != "completed":
        raise RuntimeError(f"Job {job_id} failed with state: {status.job_state}")

    job.download_outputs(output_dir=output_dir)
    result_file = os.path.join(output_dir, f"{filename}.json")

    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["elapsed_seconds"] = elapsed
    return data


def run_direct_translation(client: SarvamAI, audio_path: str, num_speakers: int | None = 2) -> dict:
    """Method 1: Direct translation with diarization via Batch API."""
    output_dir = "data/output_direct"
    return run_batch_job(client, audio_path, mode="translate", output_dir=output_dir, num_speakers=num_speakers)


def run_two_step_translation(client: SarvamAI, audio_path: str, num_speakers: int | None = 2) -> dict:
    """Method 2: Transcribe with diarization via Batch API -> Text Translate per segment."""
    output_dir = "data/output_two_step"
    stt_data = run_batch_job(client, audio_path, mode="transcribe", output_dir=output_dir, num_speakers=num_speakers)

    detected_lang = stt_data.get("language_code") or "auto"
    print(f"\nDetected audio language: {detected_lang}")

    entries = stt_data.get("diarized_transcript", {}).get("entries", [])
    translated_entries = []

    print("\nTranslating individual diarized segments into English...")
    for idx, entry in enumerate(entries, start=1):
        orig_text = entry.get("transcript", "").strip()
        speaker = entry.get("speaker_id", "unknown")
        start = entry.get("start_time_seconds", 0.0)
        end = entry.get("end_time_seconds", 0.0)

        if orig_text:
            trans_res = client.text.translate(
                input=orig_text,
                source_language_code=detected_lang if detected_lang != "unknown" else "auto",
                target_language_code="en-IN",
            )
            eng_text = trans_res.translated_text
        else:
            eng_text = ""

        translated_entries.append({
            "speaker_id": speaker,
            "start_time_seconds": start,
            "end_time_seconds": end,
            "original_text": orig_text,
            "translated_text": eng_text,
        })
        print(f"  Segment {idx} [{speaker}]: '{orig_text}' -> '{eng_text}'")

    # Also translate full transcript for overall context
    full_orig = stt_data.get("transcript", "")
    full_trans_res = client.text.translate(
        input=full_orig,
        source_language_code=detected_lang if detected_lang != "unknown" else "auto",
        target_language_code="en-IN",
    ) if full_orig else None

    return {
        "source_data": stt_data,
        "language_code": detected_lang,
        "diarized_entries": translated_entries,
        "full_original": full_orig,
        "full_translated": full_trans_res.translated_text if full_trans_res else "",
        "elapsed_seconds": stt_data.get("elapsed_seconds", 0.0),
    }


def compare_and_print(direct_res: dict, two_step_res: dict):
    """Print a side-by-side comparison of both approaches and save summary."""
    print("\n" + "=" * 80)
    print(" " * 28 + "COMPARISON RESULTS")
    print("=" * 80)

    print("\nPerformance:")
    print(f"  • Method 1 (Direct Batch Translate) Execution Time:   {direct_res.get('elapsed_seconds', 0.0):.2f}s")
    print(f"  • Method 2 (Two-Step: STT + Text Trans) STT Time:    {two_step_res.get('elapsed_seconds', 0.0):.2f}s")

    print("\nFull Text Comparison:")
    print(f"  • Direct Translate: {direct_res.get('transcript', '')}")
    print(f"  • Two-Step Original ({two_step_res.get('language_code')}): {two_step_res.get('full_original', '')}")
    print(f"  • Two-Step Translate (en-IN): {two_step_res.get('full_translated', '')}")

    print("\nDiarized Segments Comparison:")
    direct_entries = direct_res.get("diarized_transcript", {}).get("entries", [])
    two_step_entries = two_step_res.get("diarized_entries", [])

    max_len = max(len(direct_entries), len(two_step_entries))
    for i in range(max_len):
        print(f"\n--- Segment #{i + 1} ---")
        if i < len(direct_entries):
            d = direct_entries[i]
            print(f"  [Method 1 Direct]   {d.get('speaker_id')} ({d.get('start_time_seconds', 0):.2f}s - {d.get('end_time_seconds', 0):.2f}s):")
            print(f"      English: '{d.get('transcript')}'")
        else:
            print("  [Method 1 Direct]   No matching segment")

        if i < len(two_step_entries):
            t = two_step_entries[i]
            print(f"  [Method 2 Two-Step] {t.get('speaker_id')} ({t.get('start_time_seconds', 0):.2f}s - {t.get('end_time_seconds', 0):.2f}s):")
            print(f"      Source : '{t.get('original_text')}'")
            print(f"      English: '{t.get('translated_text')}'")
        else:
            print("  [Method 2 Two-Step] No matching segment")

    # Save comparison report
    summary = {
        "direct_translation": direct_res,
        "two_step_translation": two_step_res,
    }
    os.makedirs("data/output", exist_ok=True)
    report_path = "data/output/comparison_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed JSON report saved to: {report_path}")
    print("=" * 80)


def main():
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise RuntimeError("Set SARVAM_API_KEY environment variable first.")

    client = SarvamAI(api_subscription_key=api_key)

    audio_path = sys.argv[1] if len(sys.argv) > 1 else "data/Sequentially.wav"
    num_speakers = 2

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")

    print(f"Starting comparison test for: {audio_path} (num_speakers={num_speakers})")

    # 1. Run Direct Translation via Batch API
    print("\n>>> RUNNING METHOD 1: Direct Translation via Batch API <<<")
    direct_res = run_direct_translation(client, audio_path, num_speakers=num_speakers)

    # 2. Run Two-Step Translation
    print("\n>>> RUNNING METHOD 2: Two-Step (Batch Diarized STT + Text Translation) <<<")
    two_step_res = run_two_step_translation(client, audio_path, num_speakers=num_speakers)

    # 3. Compare Both
    compare_and_print(direct_res, two_step_res)


if __name__ == "__main__":
    main()
