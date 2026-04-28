"""ADB-backed Android device integration for live TorchRoyale runs."""

import io
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREENSHOT_SIZE = (368, 652)


class AndroidDevice:
    """Minimal Android device wrapper for screenshots and taps."""

    def __init__(
        self,
        device_serial: str = "",
        ip: str = "",
        adb_path: Optional[str] = None,
        screenshot_size: tuple[int, int] = DEFAULT_SCREENSHOT_SIZE,
    ) -> None:
        self.ip = ip.strip()
        self._adb_path = adb_path or self._resolve_adb_path()
        self.device_serial = device_serial.strip()
        self.screenshot_size = screenshot_size

        self._ensure_server()
        self.device_serial = self._resolve_device_serial(self.device_serial)

    @staticmethod
    def _resolve_adb_path() -> str:
        adb_path = shutil.which("adb")
        if adb_path:
            return adb_path

        local_adb = REPO_ROOT / "platform-tools" / "adb"
        if local_adb.exists():
            return str(local_adb)

        raise FileNotFoundError(
            "ADB binary not found. Install Android platform-tools or set adb_path."
        )

    def _base_command(self) -> list[str]:
        command = [self._adb_path]
        if self.device_serial:
            command.extend(["-s", self.device_serial])
        return command

    def _run(self, args: list[str], timeout: int = 15) -> str:
        command = self._base_command() + args
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout.strip()

    def _run_bytes(self, args: list[str], timeout: int = 15) -> bytes:
        command = self._base_command() + args
        result = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        return result.stdout

    def _ensure_server(self) -> None:
        subprocess.run([self._adb_path, "start-server"], capture_output=True, check=False)
        if self.ip:
            subprocess.run(
                [self._adb_path, "connect", self.ip],
                capture_output=True,
                text=True,
                check=False,
            )

    def _resolve_device_serial(self, preferred_serial: str) -> str:
        if preferred_serial:
            try:
                subprocess.run(
                    [self._adb_path, "-s", preferred_serial, "get-state"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                return preferred_serial
            except subprocess.CalledProcessError:
                pass

        devices_output = subprocess.run(
            [self._adb_path, "devices"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        devices = [
            line.split()[0]
            for line in devices_output.splitlines()
            if "\tdevice" in line
        ]
        if not devices:
            raise RuntimeError("No connected Android devices found via adb.")
        return devices[0]

    def click(self, x: int, y: int) -> None:
        self._run(["shell", "input", "tap", str(x), str(y)])

    def start_game(self) -> None:
        self._run(
            [
                "shell",
                "am",
                "start",
                "-n",
                "com.supercell.clashroyale/com.supercell.titan.GameApp",
            ]
        )

    def stop_game(self) -> None:
        self._run(["shell", "am", "force-stop", "com.supercell.clashroyale"])

    def take_screenshot(self) -> Image.Image:
        raw = self._run_bytes(["exec-out", "screencap", "-p"], timeout=20)
        if not raw:
            raise RuntimeError("ADB returned an empty screenshot.")
        image = Image.open(io.BytesIO(raw.replace(b"\r\n", b"\n"))).convert("RGB")
        return image.resize(self.screenshot_size, Image.Resampling.BICUBIC)

    def load_deck(self, card_ids: list[int]) -> None:
        deck_ids = ";".join(str(card_id) for card_id in card_ids)
        slots = ";".join("0" for _ in card_ids)
        url = (
            "https://link.clashroyale.com/en/?clashroyale://copyDeck"
            f"?deck={deck_ids}&slots={slots}&tt=159000000&l=Royals&id=JR2RU0L90"
        )
        self._run(
            [
                "shell",
                "am",
                "start",
                "-a",
                "android.intent.action.VIEW",
                "-d",
                url,
            ]
        )
        time.sleep(2)
