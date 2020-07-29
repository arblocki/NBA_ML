
# NBA Machine Learning Model

import pymongo
from ohmysportsfeedspy import MySportsFeeds
from src.config import config
from src.data_extract import getUpcomingGameData
from src.model import predictGames
from src.mongo import updateTodayGames, updateYesterdayGames
from src.odds import getTodaySpreads
from src import RAPM
from datetime import datetime, timedelta
import pandas as pd
import simplejson as json

# Update RAPM inputs, ratings, and stint CSVs with a previous day's games
def updateRAPMFeatures(msf, season, yesterdayStr):
    newUnits, newPoints, newWeights = RAPM.extractPbpData(msf, season, yesterdayStr, '')
    RAPM.addDateStintsToCSV(season, yesterdayStr, len(newWeights))  # Add stints onto stint CSV
    currentUnits, currentPoints, currentWeights = RAPM.importPbpDataFromJSON('../features/RAPM-inputs/' + season)
    currentUnits.extend(newUnits)
    currentPoints.extend(newPoints)
    currentWeights.extend(newWeights)
    RAPM.exportPbpDataToJSON(currentUnits, currentPoints, currentWeights, '../features/RAPM-inputs/' + season)
    newRatings = RAPM.calculateRAPM(currentUnits, currentPoints, currentWeights)
    playerDict = RAPM.getPlayerNames(msf, season)
    RAPM.exportPlayerRatings(newRatings, playerDict, '../features/RAPM-ratings/' + season)


def main():

    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, config.msfPassword)
    # Create MongoDB client instance
    client = pymongo.MongoClient('mongodb+srv://' + config.mongoBlock + ':' + config.mongoBlockPW +
                                 '@nba-data.nftax.azure.mongodb.net/NBA-ML?retryWrites=true&w=majority')

    season = '2019-2020-regular'
    todayStr = datetime.now().strftime('%Y%m%d')
    yesterdayStr = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    print('Running for today\'s date: ', todayStr, ' (yesterday was ', yesterdayStr, ')', sep='')

    # Update DB and CSVs with results from yesterday's games
    output = msf.msf_get_data(feed='daily_games', league='nba', season='2019-2020-regular', date=yesterdayStr,
                              format='json', force='true')
    if len(output['games']) != 0:
        yesterdayDF = pd.read_csv('../features/gameData/today-games.csv').set_index('gameID', drop=True)
        for game in output['games']:
            # Update today-games.csv w/ final scores
            yesterdayDF.loc[game['schedule']['id'], 'awayScore'] = game['score']['awayScoreTotal']
            yesterdayDF.loc[game['schedule']['id'], 'homeScore'] = game['score']['homeScoreTotal']
            # Update w/ final score and bet result in Mongo
            updateYesterdayGames(client, season, game)
        #   Append today-games.csv onto season-games.csv
        currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
        newDF = pd.concat([currentSeasonDF, yesterdayDF])
        newDF.to_csv('../features/gameData/' + season + '-games.csv')

        # Update RAPM inputs, ratings, and stints
        updateRAPMFeatures(msf, season, yesterdayStr)

    # Get input dataframe for today's games
    gameDF = getUpcomingGameData(msf, season, todayStr)
    gameDF = predictGames(season, gameDF)
    gameDF = getTodaySpreads(gameDF)

    # Output as today-games.csv
    basePath = RAPM.getBasePath(season, 'today', '', 'gameData')
    gameDF.to_csv(basePath + '-games.csv', index=False)
    updateTodayGames(msf, client, season)

if __name__ == '__main__':
    main()