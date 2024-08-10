import logging

# Ensure the standard Python logger is used instead of AllenNLP’s custom logger
logging.setLoggerClass(logging.Logger)

logger = logging.getLogger("parser")
logger.propagate = False

fileHandler = logging.FileHandler('parser.log', 'w')
logger.addHandler(fileHandler)
logger.setLevel(logging.INFO)
