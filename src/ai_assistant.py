"""
ai_assistant.py — OpenAI API integration for tracking log QA and visual person analysis.

Provides:
- Visual attribute captioning via OpenAI Vision (clothing, colors, accessories).
- Natural language QA over multi-camera tracking logs (timestamps, counts, movement).
- Context builder that formats tracking_log.json into token-efficient system prompts.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional
from src import config


def get_openai_client(api_key: Optional[str] = None):
    """
    Safely initialize and return an OpenAI client instance.
    Raises ValueError if key is missing or openai package is unavailable.
    """
    key = api_key or config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file."
        )

    try:
        from openai import OpenAI
        return OpenAI(api_key=key)
    except ImportError:
        raise ImportError(
            "The 'openai' Python package is not installed. Please run: pip install openai"
        )


class TrackingLogContextBuilder:
    """Formats tracking log data into structured natural language for the LLM system prompt."""

    @staticmethod
    def build_system_context(log_data: dict) -> str:
        cameras = log_data.get("cameras", {})
        persons = log_data.get("persons", {})

        lines = [
            "You are an intelligent AI assistant analyzing multi-camera person tracking footage.",
            "You have access to structured tracking logs collected across synced camera feeds.",
            "",
            "=== CAMERA METADATA ===",
        ]

        for cam_id, meta in cameras.items():
            fps = meta.get("fps", 25.0)
            total_f = meta.get("total_frames", 0)
            duration_sec = total_f / fps if fps > 0 else 0
            lines.append(
                f"- Camera {cam_id}: File={meta.get('video_file', 'unknown')}, "
                f"FPS={fps:.1f}, Total Frames={total_f} (~{duration_sec:.1f}s)"
            )

        lines.extend([
            "",
            "=== DETECTED INDIVIDUALS & APPEARANCE TIMELINES ===",
        ])

        if not persons:
            lines.append("No persons registered in the tracking log.")
        else:
            for gid_str, pinfo in sorted(persons.items(), key=lambda x: int(x[0])):
                gid = pinfo.get("global_id", gid_str)
                cams_seen = pinfo.get("cameras_seen", [])
                cams_str = ", ".join(f"Cam {c}" for c in cams_seen) if cams_seen else "None"
                visual_attr = pinfo.get("visual_attributes", "Not described yet")
                total_dets = pinfo.get("total_detections", 0)

                lines.append(f"\nPerson Global ID: {gid}")
                lines.append(f"  - Cameras Seen: [{cams_str}]")
                lines.append(f"  - Total Detections: {total_dets}")
                lines.append(f"  - Clothing / Visual Attributes: {visual_attr}")
                lines.append("  - Appearance Intervals:")

                appearances = pinfo.get("appearances", [])
                if not appearances:
                    lines.append("    (No continuous appearance intervals recorded)")
                else:
                    for idx, app in enumerate(appearances, 1):
                        cam = app.get("camera_id")
                        t_start = app.get("first_timestamp", "0:00")
                        t_end = app.get("last_timestamp", "0:00")
                        f_start = app.get("first_frame", 0)
                        f_end = app.get("last_frame", 0)
                        dur = app.get("duration_seconds", 0)
                        lines.append(
                            f"    {idx}. Camera {cam}: {t_start} -> {t_end} "
                            f"(Frames {f_start}-{f_end}, duration {dur}s)"
                        )

        lines.extend([
            "",
            "=== INSTRUCTIONS FOR YOUR ANSWERS ===",
            "1. Be concise, direct, and factual based strictly on the tracking log data.",
            "2. When referring to timestamps, quote exact timestamps (e.g. 0:15) and camera IDs.",
            "3. If asked about clothing/appearance and it is provided in 'Clothing / Visual Attributes', describe it.",
            "4. If information is missing or not visible in the footage, clearly state so.",
        ])

        return "\n".join(lines)


def describe_person_thumbnail(
    thumbnail_b64: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Calls OpenAI Vision API (gpt-4o-mini by default) to generate a concise description
    of a person crop image (clothing, colors, accessories, gender presentation).
    """
    client = get_openai_client(api_key)
    model_name = model or config.OPENAI_MODEL or "gpt-4o-mini"

    # Ensure clean base64 data url
    if not thumbnail_b64.startswith("data:image"):
        image_url = f"data:image/jpeg;base64,{thumbnail_b64}"
    else:
        image_url = thumbnail_b64

    prompt = (
        "Briefly describe this person's appearance for a security tracking system. "
        "Focus on: clothing type and colors (top, bottom, outerwear), headwear, footwear, "
        "bags/accessories, and general appearance. Keep it to 1-2 concise sentences."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "low"},
                    },
                ],
            }
        ],
        max_tokens=150,
        temperature=0.2,
    )

    description = response.choices[0].message.content.strip()
    return description


def auto_caption_persons(
    log_data: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """
    Iterates through persons in log_data, generates visual descriptions for
    any person thumbnail missing 'visual_attributes', and returns updated log_data.
    """
    persons = log_data.get("persons", {})
    if not persons:
        return log_data

    client = None
    try:
        client = get_openai_client(api_key)
    except Exception as e:
        print(f"[AIAssistant] Visual captioning skipped (OpenAI client unavailable: {e})")
        return log_data

    model_name = model or config.OPENAI_MODEL or "gpt-4o-mini"
    updated_count = 0

    for gid_str, pinfo in persons.items():
        if pinfo.get("visual_attributes"):
            continue

        thumb_b64 = pinfo.get("thumbnail_b64")
        if not thumb_b64:
            continue

        try:
            desc = describe_person_thumbnail(thumb_b64, api_key=api_key, model=model_name)
            pinfo["visual_attributes"] = desc
            updated_count += 1
            print(f"[AIAssistant] Person {gid_str} visual description: {desc}")
        except Exception as err:
            print(f"[AIAssistant] Error describing Person {gid_str}: {err}")

    if updated_count > 0:
        print(f"[AIAssistant] Successfully auto-captioned {updated_count} person(s).")

    return log_data


def query_tracking_ai(
    user_prompt: str,
    log_data: dict,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Queries OpenAI API with user's question using the formatted tracking log context.
    """
    try:
        client = get_openai_client(api_key)
    except Exception as e:
        return (
            f"❌ **OpenAI API Key Missing or Invalid**\n\n"
            f"To use the AI Assistant, please set your OpenAI API key in your `.env` file:\n"
            "```env\nOPENAI_API_KEY=your_actual_api_key_here\n```\n"
            f"Error details: {e}"
        )

    model_name = model or config.OPENAI_MODEL or "gpt-4o-mini"
    system_context = TrackingLogContextBuilder.build_system_context(log_data)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as err:
        return f"❌ **OpenAI API Error**: {err}"
