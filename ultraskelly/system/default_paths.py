import time
from datetime import datetime
from pathlib import Path

DEFAULT_BASE_FOLDER_NAME = "ultraskelly_data"


def os_independent_home_dir() -> str:
    return str(Path.home())


def get_default_base_folder_path() -> str:
    return str(Path(os_independent_home_dir()) / DEFAULT_BASE_FOLDER_NAME)




def get_log_file_path() -> str:
    log_folder_path = (
            Path(get_default_base_folder_path()) / 'logs'
    )
    log_folder_path.mkdir(exist_ok=True, parents=True)
    log_file_path = log_folder_path / create_log_file_name()
    return str(log_file_path)


def get_gmt_offset_string() -> str:
    # from - https://stackoverflow.com/a/53860920/14662833
    gmt_offset_int = int(time.localtime().tm_gmtoff / 60 / 60)
    return f"{gmt_offset_int:+}"


def create_log_file_name() -> str:
    return "log_" + get_iso6201_time_string() + ".log"


def get_iso6201_time_string(timespec: str = "milliseconds", make_filename_friendly: bool = True) -> str:
    iso6201_timestamp = datetime.now().isoformat(timespec=timespec)
    gmt_offset_string = f"_gmt{get_gmt_offset_string()}"
    iso6201_timestamp_w_gmt = iso6201_timestamp + gmt_offset_string
    if make_filename_friendly:
        iso6201_timestamp_w_gmt = iso6201_timestamp_w_gmt.replace(":", "_")
        iso6201_timestamp_w_gmt = iso6201_timestamp_w_gmt.replace(".", "ms")
    return iso6201_timestamp_w_gmt


def default_recording_name(string_tag: str = "") -> str:
    if len(string_tag) > 0:
        string_tag = f"_{string_tag}"
    else:
        string_tag = ""

    return time.strftime(get_iso6201_time_string(timespec="seconds") + string_tag)
