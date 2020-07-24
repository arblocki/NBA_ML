
# NBA Machine Learning Model

from ohmysportsfeedspy import MySportsFeeds
from src.config import config
from src.data_extract import getUpcomingGameData
from src.model import predictGames
from src.RAPM import getBasePath
import pandas as pd

def main():

    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, config.msfPassword)

    season = '2019-2020-regular'

    gameDF = getUpcomingGameData(msf, season, '20200730')
    gameDF = predictGames(season, gameDF)

    currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games.csv')
    newDF = pd.concat([currentSeasonDF, gameDF])
    basePath = getBasePath(season, '', '', 'gameData')
    newDF.to_csv(basePath + '-games-test.csv', index=False) # SWITCH TO newDF ONCE WE ARE OVERWRITING DATA


if __name__ == '__main__':
    main()