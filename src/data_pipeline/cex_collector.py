#!/usr/bin/env python3
"""
CEX Data Collector for TENSOR-DEFI

Downloads OHLCV price data from Binance via CCXT.
Timeframe: Jan 2023 - Dec 2024 (2 years of hourly data)
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data collection parameters
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)
END_DATE = datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
TIMEFRAME = "1h"  # Hourly candles
BATCH_SIZE = 1000  # CCXT limit per request


class CEXCollector:
    """Collects OHLCV data from Binance."""

    def __init__(self, output_dir: Path, metadata_path: Path):
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load entity list
        self.entities = pd.read_csv(metadata_path / "target_entities.csv")

        # Initialize exchange
        self.exchange: Optional[ccxt.binance] = None

    async def init_exchange(self):
        """Initialize CCXT Binance connection."""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        await self.exchange.load_markets()
        logger.info(f"Connected to Binance, {len(self.exchange.markets)} markets available")

    async def close_exchange(self):
        """Close exchange connection."""
        if self.exchange:
            await self.exchange.close()

    async def fetch_ohlcv(self, symbol: str, binance_symbol: str) -> Optional[pd.DataFrame]:
        """Fetch all OHLCV data for a single asset."""
        if binance_symbol not in self.exchange.markets:
            logger.warning(f"{symbol}: {binance_symbol} not found on Binance")
            return None

        all_data = []
        since = int(START_DATE.timestamp() * 1000)
        end_ts = int(END_DATE.timestamp() * 1000)

        while since < end_ts:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    binance_symbol,
                    timeframe=TIMEFRAME,
                    since=since,
                    limit=BATCH_SIZE
                )

                if not ohlcv:
                    break

                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1  # Next candle after last

                # Small delay for rate limiting
                await asyncio.sleep(0.1)

            except ccxt.BaseError as e:
                # Narrow to CCXT's exchange/network/rate-limit hierarchy so that
                # genuine programming errors propagate instead of being swallowed.
                logger.error(f"{symbol}: Fetch error - {e}")
                break

        if not all_data:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['symbol'] = symbol

        # Filter to exact date range
        df = df[(df['timestamp'] >= START_DATE) & (df['timestamp'] <= END_DATE)]

        return df

    async def collect_single(self, symbol: str, binance_symbol: str) -> Optional[Path]:
        """Collect and save data for a single asset."""
        df = await self.fetch_ohlcv(symbol, binance_symbol)

        if df is None or df.empty:
            logger.warning(f"{symbol}: No data collected")
            return None

        # Save to parquet (efficient storage)
        output_path = self.output_dir / f"{symbol.lower()}_ohlcv.parquet"
        df.to_parquet(output_path, index=False)

        logger.info(f"{symbol}: {len(df)} candles saved")
        return output_path

    async def collect_all(self, max_concurrent: int = 3):
        """Collect all assets with rate limiting."""
        await self.init_exchange()

        try:
            # Sequential collection with progress bar (Binance rate limits)
            results = []
            for _, row in tqdm(
                self.entities.iterrows(),
                total=len(self.entities),
                desc="Collecting market data"
            ):
                result = await self.collect_single(row['symbol'], row['binance_symbol'])
                results.append((row['symbol'], result))

            # Summary
            collected = sum(1 for _, r in results if r is not None)
            self._save_summary(results)

            print(f"\n{'='*50}")
            print(f"MARKET DATA COLLECTION SUMMARY")
            print(f"{'='*50}")
            print(f"Total entities:     {len(self.entities)}")
            print(f"Successfully collected: {collected}")
            print(f"Missing:            {len(self.entities) - collected}")
            print(f"Timeframe:          {START_DATE.date()} to {END_DATE.date()}")
            print(f"Granularity:        {TIMEFRAME}")
            print(f"{'='*50}")

        finally:
            await self.close_exchange()

    def _save_summary(self, results: list):
        """Save collection summary."""
        summary = []
        for symbol, path in results:
            if path:
                df = pd.read_parquet(path)
                summary.append({
                    'symbol': symbol,
                    'file': str(path),
                    'rows': len(df),
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat(),
                    'missing_pct': 0  # TODO: Calculate gaps
                })
            else:
                summary.append({
                    'symbol': symbol,
                    'file': None,
                    'rows': 0,
                    'start': None,
                    'end': None,
                    'missing_pct': 100
                })

        summary_df = pd.DataFrame(summary)
        summary_path = self.output_dir.parent / "metadata" / "market_data_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved summary: {summary_path}")

    def load_all(self) -> pd.DataFrame:
        """Load all collected data into a single DataFrame."""
        dfs = []
        for parquet_file in self.output_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)

        if not dfs:
            raise ValueError("No data files found")

        return pd.concat(dfs, ignore_index=True)


async def main():
    """Run market data collection."""
    base_path = Path(__file__).parent.parent.parent
    collector = CEXCollector(
        output_dir=base_path / "data" / "market",
        metadata_path=base_path / "data" / "metadata"
    )
    await collector.collect_all()


if __name__ == "__main__":
    asyncio.run(main())
