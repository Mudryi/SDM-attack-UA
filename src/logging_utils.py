import logging


def info(msg: str):
    logging.info("\t" + msg + "...")


def finish():
    logging.info("'============finish=========='")
