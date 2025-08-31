# bot.py
from time import sleep
import logging
import config
import logger as logmod
from price_fetcher import PriceFetcher
import signal
import sys

LOG = logmod.get_logger()

# Graceful shutdown
STOP = False
def _sigterm_handler(signum, frame):
    global STOP
    STOP = True
    LOG.info("Received shutdown signal, stopping...")

signal.signal(signal.SIGINT, _sigterm_handler)
signal.signal(signal.SIGTERM, _sigterm_handler)

def main_loop(interval_seconds: int = 60):
    LOG.info("Starting bot main loop")
    pf = PriceFetcher(interval=interval_seconds)

    try:
        while not STOP:
            try:
                pf.cycle_once()
            except Exception as e:
                LOG.error(f"Error during cycle: {e}", exc_info=config.PRINT_STACK_TRACES)
            # polite sleep (PriceFetcher enforces pacing internally)
            for _ in range(int(max(1, interval_seconds / 2))):
                if STOP:
                    break
                sleep(0.5)
    finally:
        LOG.info("Shutting down. Saving state if necessary.")
        # PriceFetcher uses modules that persist state; nothing else required.

if __name__== "__main__":
    interval = 60
    try:
        interval = int(sys.argv[1]) if len(sys.argv) > 1 else interval
    except Exception:
        pass
    main_loop(interval_seconds=interval)