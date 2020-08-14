
# NBA Machine Learning Model

import pymongo
from ohmysportsfeedspy import MySportsFeeds
from src.config import config
from src.data_extract import getUpcomingGameData
from src.model import predictGames
from src.mongo import updateTodayGames, updateYesterdayGame
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
    newRatings = RAPM.calculateRAPM(currentUnits, currentPoints, currentWeights)
    playerDict = RAPM.getPlayerNames(msf, season)
    RAPM.exportPbpDataToJSON(currentUnits, currentPoints, currentWeights, '../features/RAPM-inputs/' + season)
    RAPM.exportPlayerRatings(newRatings, playerDict, '../features/RAPM-ratings/' + season)


# Update the fourFactorsInputs with a pervious day's games
def updateFourFactorsInputs(season, boxscoreData):
    fourFactorDF = pd.read_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv').set_index('id', drop=True)
    awayID = boxscoreData['game']['awayTeam']['id']
    awayStats = boxscoreData['stats']['away']['teamStats'][0]
    homeID = boxscoreData['game']['homeTeam']['id']
    homeStats = boxscoreData['stats']['home']['teamStats'][0]

    # Add stats from this game onto the existing data
    fourFactorDF.loc[awayID, 'OppFG'] += homeStats['fieldGoals']['fgMade']
    fourFactorDF.loc[awayID, 'Opp3P'] += homeStats['fieldGoals']['fg3PtMade']
    fourFactorDF.loc[awayID, 'OppFGA'] += homeStats['fieldGoals']['fgAtt']
    fourFactorDF.loc[awayID, 'OppFT'] += homeStats['freeThrows']['ftMade']
    fourFactorDF.loc[awayID, 'OppFTA'] += homeStats['freeThrows']['ftAtt']
    fourFactorDF.loc[awayID, 'OppORB'] += homeStats['rebounds']['offReb']
    fourFactorDF.loc[awayID, 'OppDRB'] += homeStats['rebounds']['defReb']
    fourFactorDF.loc[awayID, 'OppTOV'] += homeStats['defense']['tov']

    fourFactorDF.loc[homeID, 'OppFG'] += awayStats['fieldGoals']['fgMade']
    fourFactorDF.loc[homeID, 'Opp3P'] += awayStats['fieldGoals']['fg3PtMade']
    fourFactorDF.loc[homeID, 'OppFGA'] += awayStats['fieldGoals']['fgAtt']
    fourFactorDF.loc[homeID, 'OppFT'] += awayStats['freeThrows']['ftMade']
    fourFactorDF.loc[homeID, 'OppFTA'] += awayStats['freeThrows']['ftAtt']
    fourFactorDF.loc[homeID, 'OppORB'] += awayStats['rebounds']['offReb']
    fourFactorDF.loc[homeID, 'OppDRB'] += awayStats['rebounds']['defReb']
    fourFactorDF.loc[homeID, 'OppTOV'] += awayStats['defense']['tov']

    fourFactorDF.to_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv')


# Update gameData and features with data from yesterday's games
#   Update CSVs with final scores
#   Update Mongo with final scores and bet outcomes
#   Update RAPM inputs, ratings, and stints
#   Update Four Factor inputs
def updateYesterdayData(msf, client, season):
    yesterdayStr = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    print('Running for yesterday\'s date: ', yesterdayStr, sep='')

    output = msf.msf_get_data(feed='daily_games', league='nba', season='2019-2020-regular', date=yesterdayStr,
                              format='json', force='true')
    if len(output['games']) != 0:
        print('Analyzing ', len(output['games']), ' games from yesterday', sep='')
        yesterdayDF = pd.read_csv('../features/gameData/today-games.csv').set_index('gameID', drop=True)
        for game in output['games']:
            gameID = game['schedule']['id']
            # Update today-games.csv w/ final scores
            yesterdayDF.loc[gameID, 'awayScore'] = game['score']['awayScoreTotal']
            yesterdayDF.loc[gameID, 'homeScore'] = game['score']['homeScoreTotal']
            # Update w/ final score and bet result in Mongo
            updateYesterdayGame(client, season, game)
            # Update 4 Factors inputs with data from this game
            boxscoreData = msf.msf_get_data(feed='game_boxscore', league='nba', season=season, game=gameID, format='json', force='true')
            updateFourFactorsInputs(season, boxscoreData)
        #   Append today-games.csv onto season-games.csv
        currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
        newDF = pd.concat([currentSeasonDF, yesterdayDF])
        newDF.to_csv('../features/gameData/' + season + '-games.csv')

        # Update RAPM inputs, ratings, and stints
        updateRAPMFeatures(msf, season, yesterdayStr)
    else:
        print('No games to analyze from yesterday')


def predictTodayGames(msf, client, season):
    todayStr = datetime.now().strftime('%Y%m%d')
    print('Running for today\'s date: ', todayStr, sep='')

    # Get input dataframe for today's games
    # print('Getting upcoming game data...')
    # gameDF = getUpcomingGameData(msf, season, todayStr)
    print('Training models and predicting today\'s games')
    gameDF = pd.read_csv('../features/gameData/today-games.csv')
    gameDF = predictGames(season, gameDF)
    print('Getting today\'s spreads')
    gameDF = getTodaySpreads(gameDF)

    # Output as today-games.csv
    basePath = RAPM.getBasePath(season, 'today', '', 'gameData')
    gameDF.to_csv(basePath + '-games.csv', index=False)
    print('Updating MongoDB with predScores and spread')
    updateTodayGames(msf, client, season)


# Get spreads from Odds API, and update them in CSVs, then MongoDB
def updateTodaySpreads(msf, client, season):
    gameDF = pd.read_csv('../features/gameData/today-games.csv').set_index('gameID', drop=False)
    gameDF = getTodaySpreads(gameDF)
    gameDF.to_csv('../features/gameData/today-games.csv', index=False)
    updateTodayGames(msf, client, season)


def main():
    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, config.msfPassword)
    # Create MongoDB client instance
    client = pymongo.MongoClient('mongodb+srv://' + config.mongoBlock + ':' + config.mongoBlockPW +
                                 '@nba-data.nftax.azure.mongodb.net/NBA-ML?retryWrites=true&w=majority')

    season = '2019-2020-regular'

    # updateYesterdayData(msf, client, season)
    predictTodayGames(msf, client, season)

    # updateTodaySpreads(msf, client, season)

if __name__ == '__main__':
    main()
