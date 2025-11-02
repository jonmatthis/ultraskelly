import asyncio
import json
import logging

from fastapi import FastAPI
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from ultraskelly.system.logging_configuration.handlers.websocket_log_queue_handler import get_websocket_log_queue, \
    MIN_LOG_LEVEL_FOR_WEBSOCKET
from ultraskelly.ultraskelly_app.ultraskelly_application import UltraSkellyApp, get_ultraskelly_app
from ultraskelly.utilities.wait_functions import await_10ms

logger = logging.getLogger(__name__)

BACKPRESSURE_WARNING_THRESHOLD: int = 100  # Number of frames before we warn about backpressure


class WebsocketServer:
    def __init__(self, fast_api_app: FastAPI, websocket: WebSocket):
        self.websocket = websocket
        self._app: UltraSkellyApp = get_ultraskelly_app()

        self._websocket_should_continue = True
        self.ws_tasks: list[asyncio.Task] = []

    async def __aenter__(self):
        logger.debug("Entering WebsocketRunner context manager...")
        self._websocket_should_continue = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.debug("WebsocketRunner context manager exiting...")
        self._websocket_should_continue = False

        # Only close if still connected
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
        # Cancel all tasks
        for task in self.ws_tasks:
            if not task.done():
                task.cancel()
        logger.debug("WebsocketRunner context manager exited.")

    @property
    def should_continue(self):
        return (
                 self._websocket_should_continue
                and self.websocket.client_state == WebSocketState.CONNECTED
        )

    async def run(self):
        logger.info("Starting websocket runner...")
        self.ws_tasks = [asyncio.create_task(self._logs_relay(), name="WebsocketLogsRelay"),
                         asyncio.create_task(self._client_message_handler(), name="WebsocketClientMessageHandler")]

        try:
            await asyncio.gather(*self.ws_tasks, return_exceptions=True)
        except Exception as e:
            logger.exception(f"Error in websocket runner: {e.__class__}: {e}")
            # Cancel all tasks when exiting
            for task in self.ws_tasks:
                if not task.done():
                    task.cancel()
            raise


    async def _logs_relay(self, ws_log_level: int = MIN_LOG_LEVEL_FOR_WEBSOCKET):
        logger.info("Starting websocket log relay listener...")
        logs_queue = get_websocket_log_queue()
        try:
            while self.should_continue:
                if not logs_queue.empty() and self.websocket.client_state == WebSocketState.CONNECTED:
                    log_record: logging.LogRecord = logs_queue.get_nowait()
                    if log_record.levelno < ws_log_level:
                        continue  # Skip logs below the specified level

                    # if traceback is present, replace with string
                    if log_record.exc_info:
                        log_record.exc_text = logging.Formatter().formatException(log_record.exc_info)
                        log_record.exc_info = None
                    await self.websocket.send_json(log_record)
                else:
                    await await_10ms()
        except asyncio.CancelledError:
            logger.debug("Log relay task cancelled")
        except WebSocketDisconnect:
            logger.info("Client disconnected, ending log relay task...")
        except Exception as e:
            logger.exception(f"Error in websocket log relay: {e.__class__}: {e}")
            raise

    async def _client_message_handler(self):
        """
        Handle messages from the client.
        """
        logger.info("Starting client message handler...")
        try:
            while self.should_continue:
                message = await self.websocket.receive()
                if message:
                    if "text" in message:
                        text_content = message.get("text", "")
                        # Try to parse as JSON if it looks like JSON
                        if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                            try:
                                data = json.loads(text_content)
                                logger.debug(f"Websocket received JSON message: {json.dumps(data,indent=2)}")


                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to decode JSON message: {e}")
                        else:
                            # Handle plain text messages
                            if text_content.startswith("ping"):
                                await self.websocket.send_text("pong")
                            elif text_content.startswith("pong"):
                                pass
                            else:
                                logger.info(f"Websocket received message: `{text_content}`")
                    elif "websocket" in message:
                        logger.trace(f"Received unknown websocket control message: {message}")
                    else:
                        logger.warning(f"Received unexpected message format: {message}")

        except asyncio.CancelledError:
            logger.debug("Client message handler task cancelled")
        except Exception as e:
            logger.exception(f"Error handling client message: {e.__class__}: {e}")
            raise
        finally:
            logger.info("Ending client message handler...")
