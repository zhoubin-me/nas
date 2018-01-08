import logging
from logging import handlers
import sys

log = logging.getLogger('')
log.setLevel(logging.DEBUG)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
log.addHandler(ch)

fh = logging.FileHandler('asldjf.log')
fh.setFormatter(format)
log.addHandler(fh)

logging.info('asldjflaksdf')
