"""
ai_assistant.py — OpenAI-powered Video Intelligence & Query Engine for Sensen.

Combines:
  1. Spatiotemporal tracking log (cameras, appearances, timestamps, frequencies).
  2. Multi-thumbnail visual profiling via GPT-4o-mini Vision with local caching.
  3. Interactive natural language chat with timestamp citation links: [Cam X @ M:SS].
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import dotenv

# Load .env from project root
dotenv.load_dotenv(override=True)


# Increment when the visual-profiling prompt or evidence policy changes. Older
# cached descriptions are ignored so past hallucinations do not live forever.
PROFILE_VERSION = 3


class AIAssistant:
    """
    Intelligent assistant capable of answering natural language queries about
    multi-camera tracking footage using OpenAI models (GPT-4o-mini).
    """

    VISUAL_KEYWORDS = (
        "cloth", "wear", "shirt", "pant", "jacket", "hoodie", "hat", "cap",
        "backpack", "bag", "color", "red", "blue", "green", "black", "white",
        "yellow", "dress", "look like", "appearance", "describe",
    )

    def __init__(self, tracking_data: dict, output_dir: Path):
        self.tracking_data = tracking_data
        self.output_dir = Path(output_dir)
        self.api_key: Optional[str] = os.getenv("OPENAI_API_KEY", "").strip()
        self.model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

        # Cache file for visual attributes extracted via Vision
        self.profile_cache_path = self.output_dir / "ai_person_profiles.json"
        self._profiles: Dict[str, dict] = self._load_profiles()

        # Chat history for multi-turn conversation
        self.conversation_history: List[dict] = []

    def reload_config(self) -> None:
        """Reload .env in case user updated the API key while UI was open."""
        dotenv.load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    def is_configured(self) -> bool:
        """Check if a valid non-placeholder OpenAI API key is set."""
        self.reload_config()
        if not self.api_key:
            return False
        if "your_openai_api_key_here" in self.api_key or len(self.api_key) < 15:
            return False
        return True

    def _get_client(self):
        """Create and return an OpenAI client instance."""
        from openai import OpenAI
        return OpenAI(api_key=self.api_key)

    # ── Visual Profiling Cache ────────────────────────────────────────────────

    @staticmethod
    def _get_thumbnails(person_info: Optional[dict]) -> List[str]:
        """Return up to three valid, unique thumbnails with legacy fallback."""
        if not person_info:
            return []

        candidates = person_info.get("thumbnails_b64")
        if not isinstance(candidates, list) or not candidates:
            candidates = [person_info.get("thumbnail_b64")]

        thumbnails: List[str] = []
        seen = set()
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate or candidate in seen:
                continue
            try:
                base64.b64decode(candidate, validate=True)
            except (ValueError, TypeError):
                continue
            seen.add(candidate)
            thumbnails.append(candidate)
            if len(thumbnails) == 3:
                break
        return thumbnails

    @staticmethod
    def _thumbnail_set_hash(thumbnails_b64: List[str]) -> Optional[str]:
        """Fingerprint the ordered set of images used to produce a profile."""
        if not thumbnails_b64:
            return None
        digest = hashlib.sha256()
        for thumbnail_b64 in thumbnails_b64:
            try:
                image_bytes = base64.b64decode(thumbnail_b64, validate=True)
            except (ValueError, TypeError):
                return None
            digest.update(len(image_bytes).to_bytes(8, "big"))
            digest.update(image_bytes)
        return digest.hexdigest()

    def _load_profiles(self) -> Dict[str, dict]:
        """Load only profiles that match the latest tracking-log thumbnails."""
        if self.profile_cache_path.exists():
            try:
                with open(self.profile_cache_path, "r", encoding="utf-8") as f:
                    profiles = json.load(f)
                current_persons = self.tracking_data.get("persons", {})
                valid_profiles: Dict[str, dict] = {}
                for gid, profile in profiles.items():
                    gid_str = str(gid)
                    person = current_persons.get(gid_str) or current_persons.get(gid)
                    current_thumbnails = self._get_thumbnails(person)
                    current_hash = self._thumbnail_set_hash(current_thumbnails)
                    if (
                        profile.get("profile_version") == PROFILE_VERSION
                        and current_hash is not None
                        and profile.get("thumbnail_hash") == current_hash
                    ):
                        valid_profiles[gid_str] = profile
                return valid_profiles
            except Exception as e:
                print(f"[AIAssistant] Could not read profile cache: {e}")
        return {}

    def _save_profiles(self) -> None:
        """Save cached visual profiles to JSON."""
        try:
            self.profile_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.profile_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._profiles, f, indent=2)
        except Exception as e:
            print(f"[AIAssistant] Could not save profile cache: {e}")

    def profile_person(self, gid: int, client=None) -> Optional[dict]:
        """
        Use GPT-4o-mini Vision to describe clothing, colors, and accessories
        from up to three person crops. Results are cached locally.
        """
        gid_str = str(gid)
        persons = self.tracking_data.get("persons", {})
        person_info = persons.get(gid_str) or persons.get(gid)
        if not person_info:
            self._profiles.pop(gid_str, None)
            return None

        thumbnails_b64 = self._get_thumbnails(person_info)
        thumbnail_hash = self._thumbnail_set_hash(thumbnails_b64)
        if thumbnail_hash is None:
            self._profiles.pop(gid_str, None)
            return None

        cached = self._profiles.get(gid_str)
        if cached and cached.get("thumbnail_hash") == thumbnail_hash:
            return cached
        self._profiles.pop(gid_str, None)

        if not self.is_configured():
            return None

        if client is None:
            client = self._get_client()

        prompt = (
            "These are up to three surveillance crops of the same tracked person, possibly "
            "from different cameras and viewpoints. Compare all supplied crops and produce one "
            "combined description. Describe only details that are clearly visible and consistent. "
            "Do not infer hidden clothing, bags, or accessories from shadows, folds, pose, "
            "or what a person might commonly carry. If a region is unclear or outside the "
            "crops, say that it is unclear. If the crops disagree, state the uncertainty rather "
            "than choosing one. Report upper clothing, lower clothing, footwear, "
            "hair, and only accessories supported by unambiguous visible evidence. Explicitly "
            "say 'no clearly visible bag or backpack' when none is visible in any crop. Keep the result to "
            "one or two factual sentences and avoid guessing."
        )

        try:
            content: List[dict] = [{"type": "text", "text": prompt}]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{thumbnail_b64}",
                        "detail": "high",
                    },
                }
                for thumbnail_b64 in thumbnails_b64
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_tokens=150,
                temperature=0
            )
            description = response.choices[0].message.content.strip()
            profile = {
                "global_id": gid,
                "description": description,
                "profiled_at": datetime.now().isoformat(),
                "profile_version": PROFILE_VERSION,
                "thumbnail_hash": thumbnail_hash,
                "thumbnail_count": len(thumbnails_b64),
            }
            self._profiles[gid_str] = profile
            self._save_profiles()
            return profile
        except Exception as e:
            print(f"[AIAssistant] Failed to profile Person {gid}: {e}")
            return None

    def profile_all_persons(self, progress_cb: Optional[Callable[[int, int], None]] = None) -> None:
        """Extract visual profiles for all persons in tracking data (cached)."""
        if not self.is_configured():
            return

        client = self._get_client()
        persons = self.tracking_data.get("persons", {})
        total = len(persons)
        count = 0

        for gid_str in persons.keys():
            try:
                gid = int(gid_str)
                if gid_str not in self._profiles:
                    self.profile_person(gid, client=client)
                count += 1
                if progress_cb:
                    progress_cb(count, total)
            except Exception as e:
                print(f"[AIAssistant] Error profiling {gid_str}: {e}")

    @classmethod
    def is_visual_query(cls, user_query: str) -> bool:
        """Return whether answering the query requires visual profile evidence."""
        query = user_query.lower()
        return any(keyword in query for keyword in cls.VISUAL_KEYWORDS)

    def get_visual_evidence(self, user_query: str, reply: str) -> List[dict]:
        """Return the exact person crops supporting a visual answer."""
        if not self.is_visual_query(user_query):
            return []

        requested_ids = {
            int(match.group(1))
            for match in re.finditer(
                r"(?:person|id)\s*:?[\s-]*0*(\d+)", user_query, re.IGNORECASE
            )
        }
        evidence_ids = sorted(requested_ids) if requested_ids else parse_person_ids(reply)
        persons = self.tracking_data.get("persons", {})
        evidence: List[dict] = []
        for gid in evidence_ids[:3]:
            gid_str = str(gid)
            person = persons.get(gid_str) or persons.get(gid)
            profile = self._profiles.get(gid_str)
            thumbnails = self._get_thumbnails(person)
            if not profile or not thumbnails:
                continue
            evidence.append({
                "global_id": gid,
                "thumbnails_b64": thumbnails,
                "description": profile.get("description", ""),
                "profiled_at": profile.get("profiled_at"),
            })
        return evidence

    def get_cached_profile(self, gid: int) -> Optional[dict]:
        """Return a copy of the current-image visual profile, if available."""
        profile = self._profiles.get(str(gid))
        return dict(profile) if profile else None

    # ── Knowledge Representation ──────────────────────────────────────────────

    def build_system_context(self) -> str:
        """
        Compile tracking data, timelines, and visual descriptions into a structured prompt context.
        """
        cameras = self.tracking_data.get("cameras", {})
        persons = self.tracking_data.get("persons", {})

        lines = [
            "# Sensen Multi-Camera Tracking Knowledge Base",
            f"Total Cameras: {len(cameras)}",
            f"Total Unique Persons Tracked: {len(persons)}",
            "",
            "## Cameras Information:"
        ]

        for cid, cinfo in sorted(cameras.items(), key=lambda x: int(x[0])):
            fps = cinfo.get("fps", 25.0)
            total_f = cinfo.get("total_frames", 0)
            dur = round(total_f / fps, 1) if fps > 0 else 0
            lines.append(f"- Camera {cid}: {dur}s total duration ({total_f} frames @ {fps} fps)")

        lines.append("")
        lines.append("## Persons Summary & Detailed Timelines:")

        for gid_str, pinfo in sorted(persons.items(), key=lambda x: int(x[0])):
            gid = int(gid_str)
            cams = pinfo.get("cameras_seen", [])
            total_dets = pinfo.get("total_detections", 0)
            appearances = pinfo.get("appearances", [])

            profile = self._profiles.get(gid_str, {})
            clothing_desc = profile.get("description", "Not yet visually analyzed.")

            app_summaries = []
            for a in appearances:
                cid = a.get("camera_id")
                t1 = a.get("first_timestamp", "0:00")
                t2 = a.get("last_timestamp", "0:00")
                dur = a.get("duration_seconds", 0)
                app_summaries.append(f"Cam {cid} [{t1} to {t2}, {dur}s]")

            app_str = "; ".join(app_summaries) if app_summaries else "No logged continuous appearances"

            lines.append(f"### [Person {gid}]")
            lines.append(f"- Cameras Visited: {cams}")
            lines.append(f"- Total Frame Sightings: {total_dets}")
            lines.append(f"- Timeline Sightings: {app_str}")
            lines.append(f"- Visual Profile & Clothing: {clothing_desc}")
            lines.append("")

        return "\n".join(lines)

    # ── Interactive Query Engine ──────────────────────────────────────────────

    def ask_async(
        self,
        user_query: str,
        callback: Callable[[str, Optional[Exception]], None]
    ) -> None:
        """
        Submit a natural language question asynchronously to avoid blocking the UI thread.
        Calls callback(response_text, error) when completed.
        """
        def _worker():
            try:
                response_text = self.ask(user_query)
                callback(response_text, None)
            except Exception as e:
                callback("", e)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def ask(self, user_query: str) -> str:
        """
        Synchronously answer a query using OpenAI Chat Completions.
        """
        self.reload_config()
        if not self.is_configured():
            return (
                "⚠️ **OpenAI API Key Not Found or Not Configured**\n\n"
                "Please add your API key to the `.env` file in the project root:\n"
                "```env\nOPENAI_API_KEY=sk-...\n```\n"
                "Then send your question again!"
            )

        client = self._get_client()

        # Check if user query asks about appearance/clothing/accessories
        if self.is_visual_query(user_query):
            # Profile only explicitly requested IDs. Reverse appearance searches
            # still need profiles for everyone, but "what is ID 0 wearing?" does not.
            persons = self.tracking_data.get("persons", {})
            requested_ids = {
                int(match.group(1))
                for match in re.finditer(
                    r"(?:person|id)\s*:?[\s-]*0*(\d+)", user_query, re.IGNORECASE
                )
            }
            gids_to_profile = (
                [str(gid) for gid in requested_ids if str(gid) in persons]
                if requested_ids
                else list(persons.keys())
            )
            for gid_str in gids_to_profile:
                if gid_str not in self._profiles:
                    try:
                        self.profile_person(int(gid_str), client=client)
                    except Exception as e:
                        print(f"[AIAssistant] On-demand profile error for {gid_str}: {e}")

        system_context = self.build_system_context()

        system_instructions = (
            "You are Sensen AI Video Intelligence Assistant, analyzing multi-camera surveillance footage.\n"
            "You have full tracking knowledge (spatiotemporal appearance logs, timestamps, camera IDs, and visual clothing profiles).\n\n"
            "Formatting & Citation Rules:\n"
            "1. Whenever you mention a specific camera sighting or time, ALWAYS format it as `[Cam X @ M:SS]` (e.g. `[Cam 1 @ 0:14]`). "
            "The user interface renders this as a clickable link that jumps directly to that video moment.\n"
            "2. Whenever you mention a person ID, format it as `[Person X]` (e.g. `[Person 0]`).\n"
            "3. Answer questions clearly, accurately, and concisely based on the provided tracking knowledge.\n"
            "4. For questions like 'where was Person X and what time?', list their camera trajectory and timestamps chronologically.\n"
            "5. For questions like 'what clothes is he wearing?', use the visual profile descriptions.\n"
            "6. For reverse search questions like 'who wore a black jacket?', scan the visual profiles and identify the matching person(s).\n"
            "7. If something was not captured or is uncertain, state it honestly.\n"
            "8. Never output raw LaTeX syntax (such as \\textbf{}, \\frac{}, $, or $$). Use plain words, clean Unicode math symbols (≈, ×, %, ≤, ≥), and bullet points (•).\n"
            "9. Avoid excessive markdown double asterisks (**) — format text cleanly and directly."
        )

        messages = [
            {"role": "system", "content": f"{system_instructions}\n\n{system_context}"}
        ]

        # Add recent conversation turns for context
        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": user_query})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=600
        )

        reply = response.choices[0].message.content.strip()

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_query})
        self.conversation_history.append({"role": "assistant", "content": reply})

        return reply

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()


def parse_camera_timestamp(text: str) -> List[Tuple[int, str, int]]:
    """
    Find citation occurrences of `[Cam X @ M:SS]` or `Cam X at M:SS` in a string.
    Returns list of (camera_id, timestamp_str, total_seconds).
    Example: "[Cam 2 @ 1:25]" -> [(2, "1:25", 85)]
    """
    pattern = r'(?:\[)?Cam\s*(\d+)\s*(?:@|at)\s*(\d+):(\d{2})(?:\])?'
    results = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        cam_id = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        total_seconds = minutes * 60 + seconds
        results.append((cam_id, f"{minutes}:{seconds:02d}", total_seconds))
    return results


def parse_person_ids(text: str) -> List[int]:
    """
    Find occurrences of `[Person X]` or `Person X` in a string.
    Returns list of person global IDs.
    """
    pattern = r'(?:\[)?Person\s*(\d+)(?:\])?'
    results = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        gid = int(match.group(1))
        if gid not in results:
            results.append(gid)
    return results
