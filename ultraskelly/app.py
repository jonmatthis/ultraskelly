import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse

import ultraskelly
from ultraskelly.api.http.app.ephemeral_key import ephemeral_key_router
from ultraskelly.api.http.app.health import health_router
from ultraskelly.api.http.app.shutdown import shutdown_router
from ultraskelly.api.middleware.add_middleware import add_middleware
from ultraskelly.api.middleware.cors import cors
from ultraskelly.api.routers import ULTRASKELLY_ROUTERS
from ultraskelly.api.server_constants import APP_URL
from ultraskelly.api.websocket.websocket_connect import websocket_router
from ultraskelly.system.default_paths import get_default_base_folder_path
from ultraskelly.ultraskelly_app.ultraskelly_application import create_ultraskelly_app

logger = logging.getLogger(__name__)




@asynccontextmanager
async def app_lifespan(
        app: FastAPI
) -> AsyncGenerator[None, None]:
    """
    Manage the application lifecycle.
    All startup and shutdown logic goes here.
    """
    # ===== STARTUP =====
    logger.api("UltraSkelly API starting...")

    # Ensure base folder exists
    base_path = Path(get_default_base_folder_path())
    base_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Base folder: {base_path}")
    create_ultraskelly_app()
    logger.success(
        f"UltraSkelly API {ultraskelly.__version__} started successfully 💀🤖\n"
        f"Swagger API docs: {APP_URL}/docs"
    )

    # Let the application do its thing
    yield

    # ===== SHUTDOWN =====
    logger.api("UltraSkelly API shutting down...")


    # Cleanup UltraSkelly application
    # get_ultraskelly_app().close()

    logger.success("UltraSkelly API shutdown complete - Goodbye! 👋")


def create_fastapi_app() -> FastAPI:
    # Create app with lifespan manager
    app = FastAPI(lifespan=app_lifespan)


    # Configure CORS
    cors(app)

    # Register routes
    _register_routes(app)

    # Add middleware
    add_middleware(app)

    # Customize OpenAPI
    _customize_openapi(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all application routes."""

    # Root redirect
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/docs")

    # Favicon
    # @app.get("/favicon.ico", include_in_schema=False)
    # async def favicon():
    #     return FileResponse(ULTRASKELLY_FAVICON_ICO_PATH)
    logger.api(f"\nRegistering WebSocket routes:")
    app.include_router(websocket_router)
    for route in websocket_router.routes:
        logger.api(f"\tRegistered WebSocket route: {route.path}")
    # Health and shutdown routes (no prefix)
    logger.api("\nRegistering App level routes:")
    for router in [health_router, shutdown_router, ephemeral_key_router]:
        app.include_router(router)
        for route in router.routes:
            logger.api(f"\tRegistered: {route.path} with methods: [{', '.join(route.methods)}]")

    logger.api("\nRegistering UltraSkelly endpoints:")
    for router in ULTRASKELLY_ROUTERS:
        app.include_router(router, prefix=f"/{ultraskelly.__package_name__}")
        for route in router.routes:
            logger.api(f"\tRegistering route: `/{ultraskelly.__package_name__}{route.path}` with methods: [{', '.join(route.methods)}]")





def _customize_openapi(app: FastAPI) -> None:
    """Customize the OpenAPI schema."""

    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        schema = get_openapi(
            title="UltraSkelly API 💀🤖✨",
            version=ultraskelly.__version__,
            description=(
                f"FastAPI Backend for UltraSkelly: {ultraskelly.__description__}"
            ),
            routes=app.routes,
        )

        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi
