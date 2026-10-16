import os
import wave
from dataclasses import dataclass
from typing import List, Optional

import torch
from pyannote.audio import Pipeline


@dataclass
class SpeakerTurn:
    speaker_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)


class PyAnnoteDiarizer:
    """Speaker Diarization using pyannote/speaker-diarization-3.1."""

    def __init__(self, hf_token: Optional[str] = None, device: Optional[str] = None):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is not set. "
                "PyAnnote requires a Hugging Face token with access to "
                "pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0."
            )

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[PyAnnote] Loading pipeline 'pyannote/speaker-diarization-3.1' on {self.device}...")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=self.hf_token,
        )
        self.pipeline.to(self.device)
        print("[PyAnnote] Pipeline loaded successfully.")

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerTurn]:
        """Run diarization and return a list of speaker turns."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file '{audio_path}' does not exist.")

        params = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        if min_speakers is not None:
            params["min_speakers"] = min_speakers
        if max_speakers is not None:
            params["max_speakers"] = max_speakers

        print(f"[PyAnnote] Diarizing '{audio_path}' (params: {params})...", flush=True)
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            with ProgressHook() as hook:
                diarization = self.pipeline(audio_path, hook=hook, **params)
        except Exception:
            diarization = self.pipeline(audio_path, **params)

        # In pyannote.audio 3.2+, pipeline returns DiarizeOutput with .speaker_diarization Annotation
        annotation = getattr(diarization, "speaker_diarization", diarization)

        turns: List[SpeakerTurn] = []
        if hasattr(annotation, "itertracks"):
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                turns.append(
                    SpeakerTurn(
                        speaker_id=str(speaker),
                        start=round(turn.start, 2),
                        end=round(turn.end, 2),
                    )
                )
        elif hasattr(annotation, "__iter__"):
            for item in annotation:
                if isinstance(item, tuple) and len(item) == 2:
                    turn, speaker = item
                elif isinstance(item, tuple) and len(item) == 3:
                    turn, _, speaker = item
                else:
                    continue
                turns.append(
                    SpeakerTurn(
                        speaker_id=str(speaker),
                        start=round(turn.start, 2),
                        end=round(turn.end, 2),
                    )
                )

        print(f"[PyAnnote] Found {len(turns)} raw speaker segment(s).", flush=True)
        return turns

    @staticmethod
    def merge_adjacent_turns(
        turns: List[SpeakerTurn], max_gap: float = 0.5
    ) -> List[SpeakerTurn]:
        """Merge consecutive segments from the same speaker if the silence gap <= max_gap seconds."""
        if not turns:
            return []

        merged: List[SpeakerTurn] = [turns[0]]
        for current in turns[1:]:
            prev = merged[-1]
            if current.speaker_id == prev.speaker_id and (current.start - prev.end) <= max_gap:
                merged[-1] = SpeakerTurn(
                    speaker_id=prev.speaker_id,
                    start=prev.start,
                    end=max(prev.end, current.end),
                )
            else:
                merged.append(current)

        return merged

    @staticmethod
    def slice_wav_segment(
        input_wav: str, start_sec: float, end_sec: float, output_wav: str
    ) -> str:
        """Slice a portion of a WAV file without quality loss using standard wave module."""
        with wave.open(input_wav, "rb") as infile:
            params = infile.getparams()
            framerate = infile.getframerate()

            start_frame = int(start_sec * framerate)
            end_frame = int(end_sec * framerate)
            num_frames = max(0, end_frame - start_frame)

            infile.setpos(start_frame)
            frames = infile.readframes(num_frames)

        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        with wave.open(output_wav, "wb") as outfile:
            outfile.setparams(params)
            outfile.writeframes(frames)

        return output_wav
