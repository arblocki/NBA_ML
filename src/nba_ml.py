
# NBA Machine Learning Model

from ohmysportsfeedspy import MySportsFeeds
from src.config import config

# Create instance of MySportsFeeds API and authenticate
msf = MySportsFeeds('2.1', verbose=True)
msf.authenticate(config.MySportsFeeds_key, config.msfPassword)



