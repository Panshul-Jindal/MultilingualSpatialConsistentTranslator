import os
from dataclasses import dataclass
from typing import Optional, Tuple
from sarvamai import SarvamAI


@dataclass
class TranslatedSpeakerSegment:
    speaker_id: str
    start: float
    end: float
    duration: float
    source_language: str
    source_transcript: str
    translated_transcript: str


class SarvamTranslator:
    """Two-Stage Speech-to-Text and Text-to-Text Translation using Sarvam AI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SARVAM_API_KEY environment variable is not set. "
                "Please set it with export SARVAM_API_KEY='...'"
            )
        self.client = SarvamAI(api_subscription_key=self.api_key)

    def transcribe(
        self,
        audio_segment_path: str,
        language_code: str = "unknown",
        model: str = "saaras:v4",
    ) -> Tuple[str, str]:
        """Transcribe an audio slice in its original spoken language.

        Returns:
            Tuple of (source_transcript, detected_language_code)
        """
        if not os.path.exists(audio_segment_path):
            raise FileNotFoundError(f"Segment file '{audio_segment_path}' not found.")

        with open(audio_segment_path, "rb") as f:
            response = self.client.speech_to_text.transcribe(
                file=f,
                model=model,
                language_code=language_code,
                mode="transcribe",
            )

        transcript = (response.transcript or "").strip()
        detected_lang = response.language_code or language_code
        return transcript, detected_lang

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en-IN",
    ) -> str:
        """Translate text from source language to target language (English)."""
        if not text.strip():
            return ""

        effective_source = source_lang if source_lang not in ("unknown", None) else "auto"

        # Skip translation if source and target are the same language
        if effective_source == target_lang:
            return text.strip()

        response = self.client.text.translate(
            input=text,
            source_language_code=effective_source,
            target_language_code=target_lang,
        )
        return (response.translated_text or "").strip()

    def process_segment(
        self,
        audio_segment_path: str,
        speaker_id: str,
        start: float,
        end: float,
        language_code: str = "unknown",
        target_lang: str = "en-IN",
    ) -> TranslatedSpeakerSegment:
        """Run 2-stage STT + Translation on a single speaker's audio segment."""
        # Stage 1: Audio -> Native Spoken Text
        source_text, detected_lang = self.transcribe(
            audio_segment_path=audio_segment_path,
            language_code=language_code,
        )

        # Stage 2: Native Spoken Text -> English Translation
        translated_text = self.translate(
            text=source_text,
            source_lang=detected_lang,
            target_lang=target_lang,
        )

        duration = round(end - start, 2)
        return TranslatedSpeakerSegment(
            speaker_id=speaker_id,
            start=start,
            end=end,
            duration=duration,
            source_language=detected_lang,
            source_transcript=source_text,
            translated_transcript=translated_text,
        )
