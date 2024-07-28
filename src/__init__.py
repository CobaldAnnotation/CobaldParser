import logging

logger = logging.getLogger("parser")
logger.propagate = False

fileHandler = logging.FileHandler('parser.log', 'w')
fileHandler.setLevel(logging.INFO)
logger.addHandler(fileHandler)
