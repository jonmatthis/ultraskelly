import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UltraSkellyApp(BaseModel):
    #add stuff here later
    pass



ULTRASKELLY_APP: UltraSkellyApp | None = None


def create_ultraskelly_app() -> UltraSkellyApp:
    global ULTRASKELLY_APP
    if ULTRASKELLY_APP is None:
        ULTRASKELLY_APP = UltraSkellyApp()
    else:
        raise ValueError("UltraSkellyApp already exists!")
    return ULTRASKELLY_APP


def get_ultraskelly_app() -> UltraSkellyApp:
    global ULTRASKELLY_APP
    if ULTRASKELLY_APP is None:
        raise ValueError("UltraSkellyApp has not been created yet!")
    return ULTRASKELLY_APP
