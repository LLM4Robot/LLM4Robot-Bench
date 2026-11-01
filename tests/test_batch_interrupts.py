"""Console regressions without starting Isaac or a model client."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest

from robot_tasks.batch import console_interrupts
from robot_tasks.paths import PROJECT_ROOT


class BatchInterruptTests(unittest.TestCase):
    def test_handler_is_restored_after_interrupt(self):
        previous = signal.getsignal(signal.SIGINT)
        with self.assertRaises(KeyboardInterrupt):
            with console_interrupts():
                signal.raise_signal(signal.SIGINT)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous)

    @unittest.skipUnless(os.name == "nt", "Windows console inheritance")
    def test_inherited_ignore_flag_does_not_swallow_ctrl_c(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "interrupted.txt"
            code = f'''
import ctypes, signal, sys, time
from pathlib import Path
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
from robot_tasks.batch import console_interrupts
kernel = ctypes.WinDLL('kernel32')
kernel.SetConsoleCtrlHandler(None, True)
signal.signal(signal.SIGINT, signal.SIG_IGN)
with console_interrupts():
    try:
        if not kernel.GenerateConsoleCtrlEvent(0, 0):
            raise RuntimeError('Cannot send console event')
        for _ in range(40):
            time.sleep(.05)
    except KeyboardInterrupt:
        Path({str(result)!r}).write_text('interrupted')
        sys.exit(0)
sys.exit(5)
'''
            child = subprocess.run([sys.executable, "-c", code],
                                   creationflags=subprocess.CREATE_NEW_CONSOLE,
                                   timeout=10, check=False)
            self.assertEqual(child.returncode, 0)
            self.assertEqual(result.read_text(), "interrupted")
