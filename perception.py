from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path

from config import VLM_MODEL_PATH, apply_model_cache_policy

apply_model_cache_policy()

import mss
import pyautogui
from mlx_vlm import generate, load
from mlx_vlm.utils import load_config
from ocrmac import ocrmac
from PIL import Image


class VisualObserver:
    MAX_TOKENS = 150
    VLM_TEMPERATURE = 0.55
    REPETITION_PENALTY = 1.2
    DEFAULT_SCALE_FACTOR = 0.8
    TEMP_IMAGE_PATH = Path("ai_eye_current_view.jpg")

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or VLM_MODEL_PATH
        self.enabled = True
        self.available = True
        self.last_summary: str | None = None
        self.model = None
        self.processor = None
        self.config = None

        try:
            with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
                self.model, self.processor = load(self.model_path)
            self.config = load_config(self.model_path)
        except Exception as error:
            self.available = False
            print(f"[System] Visual model unavailable. OCR fallback enabled. ({error})")

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def capture_display(self, scale_factor: float = DEFAULT_SCALE_FACTOR) -> list[str]:
        if not self.enabled:
            return []

        try:
            mouse_x, mouse_y = pyautogui.position()
            with mss.MSS() as screen_capture:
                monitor = self._select_monitor(screen_capture.monitors, mouse_x, mouse_y)
                screen_image = screen_capture.grab(monitor)
        except Exception as error:
            print(f"[System] Screen capture failed: {error}")
            return []

        image = Image.frombytes(
            "RGB",
            (screen_image.width, screen_image.height),
            screen_image.bgra,
            "raw",
            "BGRX",
        )
        resized_image = self._resize_image(image, scale_factor)
        resized_image.save(self.TEMP_IMAGE_PATH, quality=90)
        return [str(self.TEMP_IMAGE_PATH)]

    def generate_summary(self, img_paths: list[str], verbose: bool = False) -> str:
        if not self.enabled:
            return "[My eyes are closed. I should sleep.]"
        if not img_paths:
            return "[I cannot see the screen right now.]"
        if not self.available or self.model is None or self.processor is None:
            return self._ocr_fallback_summary(img_paths)

        messages = [
            {"role": "system", "content": "You observe Dad's screen and summarize the visible context."},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._build_visual_prompt()},
                ],
            },
        ]
        formatted_prompt = self.processor.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        observer_prompt = formatted_prompt.replace("assistant", "observer")

        try:
            if verbose:
                output = generate(
                    self.model,
                    self.processor,
                    observer_prompt,
                    image=img_paths,
                    max_tokens=self.MAX_TOKENS,
                    temperature=self.VLM_TEMPERATURE,
                    repetition_penalty=self.REPETITION_PENALTY,
                    verbose=True,
                )
            else:
                with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), \
                    contextlib.redirect_stderr(devnull):
                    output = generate(
                        self.model,
                        self.processor,
                        observer_prompt,
                        image=img_paths,
                        max_tokens=self.MAX_TOKENS,
                        temperature=self.VLM_TEMPERATURE,
                        repetition_penalty=self.REPETITION_PENALTY,
                        verbose=False,
                    )
        except Exception as error:
            print(f"[System] Visual summary failed: {error}")
            return self._ocr_fallback_summary(img_paths)

        summary = output.text.strip()
        self.last_summary = summary
        return summary

    def run_standalone_loop(self, interval_sec: int = 30) -> None:
        try:
            while True:
                img_paths = self.capture_display()
                summary = self.generate_summary(img_paths)

                timestamp = time.strftime("%H:%M:%S")
                self.last_summary = f"[Model / {timestamp}] {summary}"
                print(self.last_summary)

                time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("\n[System] Observation loop terminated.")

    @staticmethod
    def get_pref_lang() -> list[str]:
        return ["ko-KR", "en-US"]

    @staticmethod
    def _select_monitor(monitors: list[dict], mouse_x: int, mouse_y: int) -> dict:
        selected_monitor = monitors[1]
        for monitor in monitors[1:]:
            in_horizontal_range = monitor["left"] <= mouse_x < monitor["left"] + monitor["width"]
            in_vertical_range = monitor["top"] <= mouse_y < monitor["top"] + monitor["height"]
            if in_horizontal_range and in_vertical_range:
                selected_monitor = monitor
                break
        return selected_monitor

    @staticmethod
    def _resize_image(image: Image.Image, scale_factor: float) -> Image.Image:
        width = int(image.width * scale_factor)
        height = int(image.height * scale_factor)
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _get_screen_text(img_paths: list[str]) -> list[str]:
        extracted_texts = []
        for img_path in img_paths:
            try:
                annotations = ocrmac.OCR(
                    img_path,
                    recognition_level="accurate",
                    language_preference=VisualObserver.get_pref_lang(),
                ).recognize()
                extracted_texts.append(" ".join(annotation[0] for annotation in annotations if annotation[0]))
            except Exception as error:
                print(f"[System] OCR failed: {error}")
                extracted_texts.append("")
        return extracted_texts

    def _ocr_fallback_summary(self, img_paths: list[str]) -> str:
        screen_texts = [text for text in self._get_screen_text(img_paths) if text]
        if not screen_texts:
            return "[I can see the screen image, but I cannot summarize it right now.]"
        joined_text = " ".join(screen_texts)
        return f"[OCR fallback] Visible text: {joined_text[:500]}"

    def _build_visual_prompt(self) -> str:
        return (
            "Briefly summarize the main activity on the screen in 2 sentences. "
            "Do not quote or include chat logs showing 'Dad' and 'Baby'. "
            "Describe the active application and general visual context only. "
            "Use Korean if the visible context is mostly Korean. "
            "Use English if the visible context is mostly English. "
            f"Allowed locales: {self.get_pref_lang()}."
        )
