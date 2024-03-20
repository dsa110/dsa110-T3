import os
import numpy as np
import glob
import time
import shutil

import logging

logfile = '/home/user/connor/GReX-T3/services/clear_disks.log'

# Configure the logger
logging.basicConfig(filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Use logging instead of print statements for your application logic
logging.info('Starting clear disks service')

disk_limit = 2000
disk_limit_bytes = 1024**3 * disk_limit # Convert GB into bytes
voltage_dir = '/hdd/data/voltages/'

while True:
    # Replace '/' with 'C:/' on Windows
    disk_usage = shutil.disk_usage('/hdd')

    logging.info(f"Total: {disk_usage.total / (1024**3):.2f} GB")
    logging.info(f"Used: {disk_usage.used / (1024**3):.2f} GB")
    logging.info(f"Free: {disk_usage.free / (1024**3):.2f} GB")

    if disk_usage.free > disk_limit_bytes:
        logging.info("We are okay on diskspace, going to sleep now")
        time.sleep(3600)
        continue
    else:
        logging.info(f"We are below {disk_limit} GB, clearing oldest files first")
        fl = glob.glob(voltage_dir + '*.nc')

        files = [f for f in os.listdir(voltage_dir) if os.path.isfile(os.path.join(voltage_dir, f))]
        sorted_files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(voltage_dir, f)))

        # Remove only 50 files at a time.
        for fn in sorted_files[:50]:
            logging.info(f"Removing {voltage_dir}/{fn}")
            os.system("rm -rf %s/%s" % (voltage_dir, fn) )

    
