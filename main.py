import time
import mss
import pyautogui
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.utils import load_config


class VisualObserver:
    """
    Handles screen capturing and visual summarization
    Designed to run independently or be called by a main control loop.
    """

    def __init__(self, model_path: str = "mlx-community/Qwen2-VL-7B-Instruct-4bit"):
        self.model_path = model_path
        self.model, self.processor = load(self.model_path)
        self.config = load_config(self.model_path)
        self.last_summary = None

    @staticmethod
    def capture_display(scale_factor: float = 1.0) -> list[str]:
        """
        Captures the current monitor, scales it by a percentage,
        saves to a temp file, and returns the file path.
        """
        mx, my = pyautogui.position()

        with mss.MSS() as sct:
            monitor = sct.monitors[1] # primary monitor
            for mon in sct.monitors[1:]:
                if mon["left"] <= mx < mon["left"] + mon["width"] and mon["top"] <= my < mon["top"] + mon["height"]:
                    monitor = mon
                    break

            sct_img = sct.grab(monitor)

            # BGRA to RGB
            img = Image.frombytes("RGB", (sct_img.width, sct_img.height),
                                  sct_img.bgra, "raw", "BGRX")

            width = int(img.width * scale_factor)
            height = int(img.height * scale_factor)
            img = img.resize((width, height), Image.Resampling.LANCZOS)

            temp_path = "ai_eye_current_view.jpg"
            img.save(temp_path, quality=90)
            return [temp_path]

    def generate_summary(self, img_paths: list[str]) -> str:
        """
        Feeds the image path to the VLM.
        """
        if len(img_paths) != 1:
            raise ValueError("Please provide exactly one image path.")

        system_instruction = (
            "You observe and take note of everything I do. Do not use any markdown syntax."
        )

        prompt = (
            "Take a look at the image of my screen, "
            "then identify and describe details, including private matters "
            "(images and text content, tab and section titles, file names, identity names). "
            "Ensure everything you mention appear on the image. "
        )

        # Too much for 7B
        #if self.last_summary:
        #    prompt += f" [NOTE] You answered me 10 seconds ago like: '{self.last_summary}'."

        messages = [
            {
                "role": "system",
                "content": system_instruction,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        formatted_prompt = self.processor.apply_chat_template(
            conversation=messages, # type: ignore
            tokenize=False, # output str
            add_generation_prompt=True
        )

        hacked_prompt = formatted_prompt.replace("assistant", "observer")
        #print(hacked_prompt)

        output = generate(
            self.model,
            self.processor, # type: ignore
            hacked_prompt, # type: ignore
            image=img_paths,
            max_tokens=800,
            temperature=0.55,
            repetition_penalty=1.2,
        )

        return output.text.strip()

    def run_standalone_loop(self, interval_sec: int = 30):
        """
        Basic loop for testing the observer module independently.
        """
        try:
            while True:
                img_paths = self.capture_display()
                summary = self.generate_summary(img_paths)

                timestamp = time.strftime("%H:%M:%S")
                print(f"[Model / {timestamp}] {summary}")

                self.last_summary = summary

                time.sleep(interval_sec)

        except KeyboardInterrupt:
            print("\nObservation loop terminated.")


if __name__ == "__main__":
    observer = VisualObserver()
    observer.run_standalone_loop(interval_sec=5)
