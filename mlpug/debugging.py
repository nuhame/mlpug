import os

from basics.logging import get_logger

from mlpug.mlpug_exceptions import MLPugException

module_logger = get_logger(os.path.basename(__file__))


def enable_pycharm_remote_debugging(remote_debug_ip: str, logger=None):
    """

    :param remote_debug_ip: "<ip>:<port>" for remote debugging using PyCharm
    :param logger

    :return:
    """

    if logger is None:
        logger = module_logger

    try:
        import pydevd_pycharm
    except Exception as e:
        raise MLPugException("Please `pip install pydevd-pycharm=<PyCharm version>` first. "
                             "Enabling of PyCharm remote debugging failed.") from e

    try:
        ip, port = remote_debug_ip.split(":")

        port = int(port)

        logger.debug(f"Enabling remote debugging on {ip}:{port} ...")

        pydevd_pycharm.settrace(ip, port=port, stdoutToServer=True, stderrToServer=True)
    except Exception as e:
        raise MLPugException("Enabling of PyCharm remote debugging failed.") from e
