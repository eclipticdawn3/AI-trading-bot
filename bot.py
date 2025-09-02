import argparse
import logger
import config
from price_fetcher import PriceFetcher

LOG = logger.get_logger()

def main():
    parser = argparse.ArgumentParser(description="Trading Bot Runner")
    parser.add_argument("--mode", choices=["demo", "live", "backtest"], default="demo",
                        help="demo=virtual trading loop, live=real orders, backtest=walk-forward sim")
    parser.add_argument("--interval", type=int, default=60, help="seconds between live/demo cycles")
    parser.add_argument("--days", type=int, default=30, help="days for backtest mode")
    args = parser.parse_args()

    pf = PriceFetcher(interval=args.interval)

    if args.mode == "backtest":
        LOG.info("Running backtest mode for %d days", args.days)
        pf.run_backtest(days=args.days)
        return

    # demo/live loops
    if args.mode == "live":
        LOG.info("Starting LIVE mode (VIRTUAL_TRADING will be ignored if True).")
        # You can optionally enforce: config.VIRTUAL_TRADING = False
    else:
        LOG.info("Starting DEMO mode (virtual trading).")

    LOG.info("Starting bot main loop")
    while True:
        pf.cycle_once()

if __name__ == "__main__":
    main()