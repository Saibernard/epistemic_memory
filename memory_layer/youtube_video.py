"""
YouTube Video Understanding for the Memory Layer.

Downloads YouTube videos and uses Gemini's multimodal generative model
to visually understand video content — not just the transcript.

The analysis produces:
  - Visual scene descriptions (what's shown on screen)
  - Key visual elements (diagrams, text overlays, demonstrations)
  - Actions and events (what happens in the video)
  - Combined with transcript for complete understanding

Requires:
  - yt-dlp   (video download)
  - google-genai (Gemini multimodal API)
  - GOOGLE_API_KEY env var

Usage:
    analyzer = YouTubeVideoAnalyzer(api_key="AIza...")
    result = analyzer.analyze("https://youtube.com/watch?v=...")
    # result contains visual_analysis, transcript, combined_text, etc.
"""

import os
import re
import time
import tempfile
from typing import Dict, Any, Optional, List


# Gemini video limits: ~1 hour for 1M context, ~2 hours for 2M context
# We cap at 15 minutes to keep costs reasonable and stay within limits
MAX_VIDEO_DURATION = 15 * 60  # 15 minutes in seconds
MAX_VIDEO_SIZE_MB = 500


_VISUAL_ANALYSIS_PROMPT = """You are analyzing a YouTube video. Provide a thorough, detailed understanding of everything shown in the video.

Structure your analysis as follows:

## Video Overview
A 2-3 sentence summary of what this video is about.

## Visual Content
Describe what is visually shown throughout the video — scenes, people, locations, on-screen text, graphics, diagrams, code, demonstrations, products, etc. Be specific and detailed.

## Key Points & Information
Extract all important facts, claims, instructions, or information presented in the video. Number each point.

## Actions & Events
Describe the sequence of what happens in the video chronologically.

## Notable Details
Any important visual details: brand names, URLs shown on screen, specific numbers/statistics displayed, text overlays, slide content, code snippets, etc.

Be as comprehensive as possible. The goal is to capture ALL information from this video so someone can answer any question about it without watching it."""

_VISUAL_WITH_TRANSCRIPT_PROMPT = """You are analyzing a YouTube video. You have both the video itself AND its transcript below.

Provide a thorough, comprehensive analysis that captures EVERYTHING from both the visual content and the spoken words.

TRANSCRIPT:
{transcript}

---

Now analyze the video visually and combine with the transcript to produce a complete understanding.

Structure your analysis as follows:

## Video Overview
A 2-3 sentence summary of what this video is about, combining visual and spoken content.

## Visual Content (what is SHOWN)
Describe what is visually shown throughout the video — scenes, people, locations, on-screen text, graphics, diagrams, code, demonstrations, products, etc. Focus on things NOT captured in the transcript.

## Key Points & Information
Extract ALL important facts, claims, instructions, data, or information from both the visual and spoken content. Number each point.

## Detailed Content
Provide a comprehensive, paragraph-by-paragraph breakdown of the video's full content — as if writing a detailed article from it. Include both what was said and what was shown.

## Notable Visual Details
Brand names, URLs, code snippets, formulas, diagrams, statistics, or any important text shown on screen that may not appear in the transcript.

Be extremely thorough. Someone should be able to answer ANY question about this video from your analysis alone."""


class YouTubeVideoAnalyzer:
    """
    Downloads and analyzes YouTube videos using Gemini's multimodal model.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model
        self._client = None

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY required for video understanding")

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def download_video(self, url: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Download a YouTube video using yt-dlp.
        Returns path to the downloaded file plus metadata.
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "yt-dlp is required for video download. "
                "Install with: pip install yt-dlp"
            )

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="yt_video_")

        output_template = os.path.join(output_dir, "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "worst[ext=mp4]/worst",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "max_filesize": MAX_VIDEO_SIZE_MB * 1024 * 1024,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            video_path = ydl.prepare_filename(info)
            if not os.path.exists(video_path):
                for ext in ["mp4", "webm", "mkv"]:
                    alt = os.path.join(output_dir, f"{info['id']}.{ext}")
                    if os.path.exists(alt):
                        video_path = alt
                        break

            if not os.path.exists(video_path):
                raise ValueError(f"Video download completed but file not found at {video_path}")

            duration = info.get("duration", 0)
            if duration > MAX_VIDEO_DURATION:
                os.remove(video_path)
                raise ValueError(
                    f"Video is {duration // 60}m {duration % 60}s long. "
                    f"Max supported: {MAX_VIDEO_DURATION // 60} minutes."
                )

            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            return {
                "path": video_path,
                "video_id": info.get("id", ""),
                "title": info.get("title", ""),
                "duration": duration,
                "file_size_mb": round(file_size_mb, 1),
                "ext": info.get("ext", "mp4"),
                "description": info.get("description", ""),
                "uploader": info.get("uploader", ""),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count", 0),
            }

    def analyze_video(
        self,
        video_path: str,
        transcript: Optional[str] = None,
    ) -> str:
        """
        Upload video to Gemini and get a comprehensive visual analysis.
        If transcript is provided, it's included for richer understanding.
        """
        client = self._get_client()

        print(f"  + Uploading video to Gemini ({os.path.getsize(video_path) / 1024 / 1024:.1f} MB)...")
        uploaded_file = client.files.upload(file=video_path)

        print(f"  + Waiting for video processing...")
        for _ in range(60):
            file_info = client.files.get(name=uploaded_file.name)
            if file_info.state.name == "ACTIVE":
                break
            time.sleep(2)
        else:
            raise ValueError("Video processing timed out (2 minutes). Try a shorter video.")

        if transcript and len(transcript) > 100:
            transcript_preview = transcript[:8000]
            prompt = _VISUAL_WITH_TRANSCRIPT_PROMPT.format(transcript=transcript_preview)
        else:
            prompt = _VISUAL_ANALYSIS_PROMPT

        print(f"  + Analyzing video with {self.model}...")
        response = client.models.generate_content(
            model=self.model,
            contents=[uploaded_file, prompt],
        )

        try:
            client.files.delete(name=uploaded_file.name)
        except Exception:
            pass

        return response.text

    def full_analyze(
        self,
        url: str,
        transcript_text: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete pipeline: download video -> analyze with Gemini -> return results.
        Optionally pass in a pre-fetched transcript.
        """
        download_info = self.download_video(url, output_dir=output_dir)
        video_path = download_info["path"]

        try:
            visual_analysis = self.analyze_video(video_path, transcript=transcript_text)
        finally:
            try:
                os.remove(video_path)
            except Exception:
                pass

        combined = ""
        if visual_analysis:
            combined += "# VISUAL ANALYSIS\n\n" + visual_analysis + "\n\n"
        if transcript_text:
            combined += "# TRANSCRIPT\n\n" + transcript_text

        return {
            "visual_analysis": visual_analysis,
            "transcript": transcript_text or "",
            "combined_text": combined,
            "title": download_info["title"],
            "video_id": download_info["video_id"],
            "duration": download_info["duration"],
            "file_size_mb": download_info["file_size_mb"],
            "uploader": download_info.get("uploader", ""),
            "description": download_info.get("description", ""),
        }
