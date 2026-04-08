from __future__ import annotations

import asyncio
import logging

from llm_tg_bot.bot import BridgeBot
from llm_tg_bot.config import load_settings


async def async_main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    bot = BridgeBot(settings)
    await bot.run()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
