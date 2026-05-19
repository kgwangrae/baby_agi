import queue
import threading


class TerminalUI:
    """
    Asynchronous terminal interface.
    Lets Dad type while the perception/reasoning loop is running.
    """

    def __init__(self) -> None:
        self.input_queue: queue.Queue[str] = queue.Queue()
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        print("\n[System] Terminal UI activated. Type your message anytime and press Enter.")

    def get_and_clear(self) -> str:
        messages: list[str] = []
        while True:
            try:
                messages.append(self.input_queue.get_nowait())
            except queue.Empty:
                break

        return "\n".join(messages)

    def _listen_loop(self) -> None:
        while True:
            try:
                user_text = input()
            except (EOFError, KeyboardInterrupt):
                break

            clean_text = user_text.strip()
            if clean_text:
                self.input_queue.put(clean_text)
