import asyncio
import logging
import  sys

import uvicorn

from ultraskelly.api.server_constants import PORT, HOSTNAME
from ultraskelly.app import create_fastapi_app
from ultraskelly.utilities.kill_process_on_port import kill_process_on_port
from ultraskelly.utilities.wait_functions import await_1s

logger = logging.getLogger(__name__)


async def main() -> None:
    server: uvicorn.Server | None = None

    try:
        # Clean up any existing process on the port
        kill_process_on_port(port=PORT)

        # Create FastAPI app
        app = create_fastapi_app()

        # Configure and create Uvicorn server
        config = uvicorn.Config(
            app=app,
            host=HOSTNAME,
            port=PORT,
            log_level="warning",
            reload=False

        )
        server = uvicorn.Server(config)

        logger.info(f"Starting server on {HOSTNAME}:{PORT}")

        # Run server (blocks until shutdown)
        await server.serve()

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        if server:
            server.should_exit = True
            await await_1s()  # Give it time to shut down gracefully

        logger.success("Done! Ultraskelly out 💀🤖🫡")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)
    else:
        sys.exit(0)