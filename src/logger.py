# import logging
# import os
# from datetime import datetime

# LOG_FILE = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# os.makedirs(logs_path, exist_ok=True)
# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     level=logging.INFO,
# )

import logging
import os
from datetime import datetime

# create logs directory
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# log file name
LOG_FILE = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# full path to log file
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    level=logging.INFO,
)
