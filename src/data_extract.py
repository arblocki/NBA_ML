# Game Data Extraction Functions

from ohmysportsfeedspy import MySportsFeeds
from src.config import config
from src import RAPM

import simplejson as json
import numpy as np
import pandas as pd
import time
from datetime import timedelta


# Given a timestamp in UTC ISO-6801 format, return a string (YYYYMMDD) of the previous date in EST
def getPreviousDay(timestamp):
    date = RAPM.convertDatetimeString(timestamp)
    estDate = date - timedelta(hours=4)
    dayBefore = estDate - timedelta(days=1)
    return dayBefore.strftime('%Y%m%d')


# Given a season, import the RAPM ratings from JSON file
def importPlayerRatings(season):
    path = '../features/RAPM-ratings/' + season + '-RAPM.json'
    with open(path) as inFile:
        ratings = json.load(inFile)
    ratingDict = {}
    for rating in ratings:
        ratingDict[rating['id']] = rating['rating']
    return ratingDict


# Given the current season, output a string of the previous season
def getPrevSeasonStr(season):
    if season == '2016-2017-regular':
        return '2015-2016-regular'
    elif season == '2017-2018-regular':
        return '2016-2017-regular'
    elif season == '2018-2019-regular':
        return '2017-2018-regular'
    elif season == '2019-2020-regular':
        return '2018-2019-regular'
    elif season == '2020-2021-regular':
        return '2019-2020-regular'
    raise ValueError('Invalid season given to getPrevSeasonStr')


# Given a team, season, and RAPM ratings, output the weightedTeamRAPM
def calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False):
    # Get MinutesPerGame for each player on the team
    output = msf.msf_get_data(feed='seasonal_player_stats', league='nba', season=season, team=teamID,
                              stats='minSecondsPerGame', format='json', force='true')
    # Map from playerID to injury status
    injuryDict = {}
    nameDict = {}
    if injuries:
        injuries = msf.msf_get_data(feed='player_injuries', league='nba', team=teamID, format='json', force='true')
        for player in injuries['players']:
            injuryDict[player['id']] = player['currentInjury']['playingProbability']
    # Record each player's minutes per game, as well as the team total (to calculate average)
    playerIDs = []
    minPerGameByPID = {}
    totalMinPerGame = 0
    for player in output['playerStatsTotals']:
        playerIDs.append(player['player']['id'])
        minutesPerGame = player['stats']['miscellaneous']['minSecondsPerGame'] / 60
        minPerGameByPID[player['player']['id']] = minutesPerGame
        totalMinPerGame += minutesPerGame
    avgMinPerGame = totalMinPerGame / len(playerIDs)
    teamTotalRAPM = 0
    # Sum every player's weightedRAPM ((minutesPerGame / teamAverageMinutesPerGame) * rating)
    numSkipped = 0
    for PID in playerIDs:
        if PID not in ratingDict:
            print('Skipping PID ', PID, ' in RAPM calculations due to lack of data', sep='')
            numSkipped += 1
            continue
        if PID in injuryDict:
            if (injuryDict[PID] == 'OUT') or (injuryDict[PID] == 'DOUBTFUL'):
                print('Skipping PID ', PID, ' in RAPM calculations due to injury', sep='')
                numSkipped += 1
                continue
            if injuryDict[PID] == 'QUESTIONABLE':
                # If a player is questionable, add half of their rating on
                rating = ratingDict[PID] * 0.5
                minPerGame = minPerGameByPID[PID]
                teamTotalRAPM += (minPerGame / avgMinPerGame) * rating
                continue
        rating = ratingDict[PID]
        minPerGame = minPerGameByPID[PID]
        teamTotalRAPM += (minPerGame / avgMinPerGame) * rating
    weightedTeamRAPM = teamTotalRAPM / (len(playerIDs) - numSkipped)

    return weightedTeamRAPM


# Given a team and season, get the weighted team RAPM
# Used for previous seasons, current seasons take a date range into account
def getPrevSeasonWeightedTeamRAPM(msf, season, teamID):
    ratingDict = importPlayerRatings(season)
    return calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False)


# Given a team and time-frame, get the weighted team RAPM
# Used for current seasons
def getTimeframeRAPM(msf, season, numStints, teamID):
    # Import RAPM inputs
    units, points, weights = RAPM.importPbpDataFromJSON(RAPM.getBasePath(season, '', '', 'RAPM-inputs'))
    ratings = RAPM.calculateRAPM(units[:numStints], points[:numStints], weights[:numStints])
    ratingDict = {}
    for rating in ratings:
        ratingDict[int(rating[0])] = rating[1]
    # Calculate RAPM from the specified window of data
    return calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False)


# Given a team's seasonal stats, output the four factors stats that we can calculate given our data (only 2 right now)
def calculateFourFactors(seasonalStats):
    teamFourFactors = []
    # Calculate effective FG%
    FG = seasonalStats['fieldGoals']['fgMade']
    threePtFG = seasonalStats['fieldGoals']['fg3PtMade']
    FGA = seasonalStats['fieldGoals']['fgAtt']
    EFG = (FG + (0.5 * threePtFG)) / FGA
    teamFourFactors.append(EFG)
    # Calculate free throw factor
    FT = seasonalStats['freeThrows']['ftMade']
    freeThrowFactor = FT / FGA
    teamFourFactors.append(freeThrowFactor)
    return teamFourFactors


# Given a team's seasonal stats, output an array of the relevant stats only
def extractBasicData(stats):
    # 'awayGamesPlayed', 'awayFg2PtAttPerGame', 'awayFg2PtMadePerGame', 'awayFg2PtPct',
    # 'awayFg3PtAttPerGame', 'awayFg3PtMadePerGame', 'awayFg3PtPct', 'awayFgAttPerGame', 'awayFgMadePerGame',
    # 'awayFgPct', 'awayFtAttPerGame', 'awayFtMadePerGame', 'awayFtPct', 'awayOffRebPerGame', 'awayDefRebPerGame',
    # 'awayRebPerGame', 'awayAstPerGame', 'awayPtsPerGame', 'awayTovPerGame', 'awayStlPerGame', 'awayBlkPerGame',
    # 'awayBlkAgainstPerGame', 'awayPtsAgainstPerGame', 'awayFoulsPerGame', 'awayFoulsDrawnPerGame',
    # 'awayFoulPersPerGame', 'awayFoulPersDrawnPerGame', 'awayPlusMinusPerGame', 'awayWinPct'
    basicData = [stats['gamesPlayed'],

                 stats['fieldGoals']['fg2PtAttPerGame'],
                 stats['fieldGoals']['fg2PtMadePerGame'],
                 stats['fieldGoals']['fg2PtPct'],
                 stats['fieldGoals']['fg3PtAttPerGame'],
                 stats['fieldGoals']['fg3PtMadePerGame'],
                 stats['fieldGoals']['fg3PtPct'],
                 stats['fieldGoals']['fgAttPerGame'],
                 stats['fieldGoals']['fgMadePerGame'],
                 stats['fieldGoals']['fgPct'],

                 stats['freeThrows']['ftAttPerGame'],
                 stats['freeThrows']['ftMadePerGame'],
                 stats['freeThrows']['ftPct'],

                 stats['rebounds']['offRebPerGame'],
                 stats['rebounds']['defRebPerGame'],
                 stats['rebounds']['rebPerGame'],

                 stats['offense']['astPerGame'],
                 stats['offense']['ptsPerGame'],

                 stats['defense']['tovPerGame'],
                 stats['defense']['stlPerGame'],
                 stats['defense']['blkPerGame'],
                 stats['defense']['blkAgainstPerGame'],
                 stats['defense']['ptsAgainstPerGame'],

                 stats['miscellaneous']['foulsPerGame'],
                 stats['miscellaneous']['foulsDrawnPerGame'],
                 stats['miscellaneous']['foulPersPerGame'],
                 stats['miscellaneous']['foulPersDrawnPerGame'],
                 stats['miscellaneous']['plusMinusPerGame'],

                 stats['standings']['winPct']]

    return basicData


def getColumnNames():
    columns = ['gameID']
    awayColumns = ['prevawayRAPM', 'awayRAPM',  # RAPM stats
                   'awayEFG', 'awayFT',  # Possible four factors calculations
                   'awayGamesPlayed', 'awayFg2PtAttPerGame', 'awayFg2PtMadePerGame', 'awayFg2PtPct',
                   'awayFg3PtAttPerGame', 'awayFg3PtMadePerGame', 'awayFg3PtPct', 'awayFgAttPerGame',
                   'awayFgMadePerGame', 'awayFgPct',
                   'awayFtAttPerGame', 'awayFtMadePerGame', 'awayFtPct',
                   'awayOffRebPerGame', 'awayDefRebPerGame', 'awayRebPerGame',
                   'awayAstPerGame', 'awayPtsPerGame', 'awayTovPerGame', 'awayStlPerGame', 'awayBlkPerGame',
                   'awayBlkAgainstPerGame', 'awayPtsAgainstPerGame', 'awayFoulsPerGame', 'awayFoulsDrawnPerGame',
                   'awayFoulPersPerGame', 'awayFoulPersDrawnPerGame', 'awayPlusMinusPerGame', 'awayWinPct']
    homeColumns = []
    for field in awayColumns:
        homeColumns.append(field.replace('away', 'home'))
    columns.extend(awayColumns)
    columns.extend(homeColumns)
    columns.extend(['awayScore', 'homeScore', 'spread', 'date', 'awayTeam', 'awayID', 'homeTeam', 'homeID'])
    return columns


# Given a season and timeframe, get a row of data for each game, output as Pandas dataframe
def getFinalGameData(msf, season, dateStart, dateEnd):
    # Use Seasonal feed to get list of (final) games between the dates
    if dateStart == '':  # If no date specified, get data from whole season
        output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json',
                                  force='true')
    elif dateEnd == '':  # If no end date specified, use Daily Games feed for startDate
        output = msf.msf_get_data(feed='daily_games', date=dateStart, league='nba', season=season, status='final',
                                  format='json', force='true')
    else:
        dateRange = 'from-' + dateStart + '-to-' + dateEnd
        output = msf.msf_get_data(feed='seasonal_games', date=dateRange, league='nba', season=season, status='final',
                                  format='json', force='true')
    if config.debug:
        print('Getting list of games...')
        print('\t', len(output['games']), ' games to analyze', sep='')
    games = []
    for game in output['games']:
        gameObject = {
            'id': game['schedule']['id'],
            'awayID': game['schedule']['awayTeam']['id'],
            'awayName': game['schedule']['awayTeam']['abbreviation'],
            'homeID': game['schedule']['homeTeam']['id'],
            'homeName': game['schedule']['homeTeam']['abbreviation'],
            'date': game['schedule']['startTime'],  # Date and time of the game in UTC ISO-6801
            'awayScore': game['score']['awayScoreTotal'],
            'homeScore': game['score']['homeScoreTotal'],
        }
        games.append(gameObject)

    # Load stintsByDate dataframe
    stintDF = pd.read_csv('../features/stintsByDate/' + season + '-stints.csv',
                          dtype={'date': str, 'numStints': int}).set_index('date', drop=False)

    # Array of lists of game data, eventually will be exported as a dataframe
    gameDataArray = []
    gameCount = 0
    for game in games:
        try:
            gameData = [game['id']]
            teamIDs = [game['awayID'], game['homeID']]
            rawGameDate = game['date']
            gameDate = RAPM.convertDatetimeString(rawGameDate)
            estGameDate = gameDate - timedelta(hours=4)
            prevDay = getPreviousDay(rawGameDate)
            if prevDay not in stintDF.index:
                print('Skipping gameID ', game['id'], ' from ', estGameDate.strftime('%Y/%m/%d'), ' (#', gameCount, ')',
                      sep='')
                gameCount += 1
                continue
            if config.debug:
                print('Analyzing gameID ', game['id'], ' from ', estGameDate.strftime('%Y/%m/%d'), ' (#', gameCount, ')',
                      sep='')
            for teamID in teamIDs:
                # Get past season RAPM weighted total
                prevSeasonStr = getPrevSeasonStr(season)
                prevSeasonWeightedTeamRAPM = getPrevSeasonWeightedTeamRAPM(msf, prevSeasonStr, teamID)
                gameData.append(prevSeasonWeightedTeamRAPM)

                # Get current season RAPM weighted total
                # TODO: Rewrite this section to only calculate timeframeRAPM every calendar date, instead of every game
                numStints = stintDF.loc[prevDay, 'numStints']   # Number of stints recorded for season until day before game
                timeframeRAPM = getTimeframeRAPM(msf, season, numStints, teamID)
                gameData.append(timeframeRAPM)

                # Get team data and use it to calculate four factors
                seasonalStats = msf.msf_get_data(feed='seasonal_team_stats', date=prevDay, team=teamID, league='nba',
                                                 season=season, format='json', force='true')
                if len(seasonalStats['teamStatsTotals']) == 0:
                    fourFactorsData = [0] * 2
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(seasonalStats['teamStatsTotals'][0]['stats'])
                    basicPerGameData = extractBasicData(seasonalStats['teamStatsTotals'][0]['stats'])
                gameData.extend(fourFactorsData)
                gameData.extend(basicPerGameData)
            dateObj = RAPM.convertDatetimeString(game['date'])
            dateStr = dateObj.strftime('%Y%m%d')
            gameData.extend([game['awayScore'], game['homeScore'], dateStr, game['awayName'], game['awayID'], game['homeName'], game['homeID']])
        except Exception as err:
            columns = getColumnNames()
            if config.debug:
                print('Error during game #', gameCount, ': ', err, sep='')
                print('\tOutputting data so far...')
            gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
            return gameDF

        gameDataArray.append(gameData)
        gameCount += 1
        time.sleep(2)

    if config.debug:
        print('Getting column names...')
    columns = getColumnNames()
    if config.debug:
        print('Outputting to dataframe...')
    gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
    return gameDF


# Given a date, build the input data for that day's games and return it as a dataframe
def getUpcomingGameData(msf, season, date):
    output = msf.msf_get_data(feed='daily_games', date=date, league='nba', season=season, format='json', force='true')
    if config.debug:
        print('Getting list of games...')
        print('\t', len(output['games']), ' games to analyze', sep='')
    games = []
    for game in output['games']:
        gameObject = {
            'id': game['schedule']['id'],
            'awayID': game['schedule']['awayTeam']['id'],
            'awayName': game['schedule']['awayTeam']['abbreviation'],
            'homeID': game['schedule']['homeTeam']['id'],
            'homeName': game['schedule']['homeTeam']['abbreviation'],
            'date': game['schedule']['startTime'],  # Date and time of the game in UTC ISO-6801
            'awayScore': -1,
            'homeScore': -1,
        }
        games.append(gameObject)

    ratingDict = importPlayerRatings(season)

    # Array of lists of game data, eventually will be exported as a dataframe
    gameDataArray = []
    gameCount = 0
    for game in games:
        try:
            gameData = [game['id']]
            teamIDs = [game['awayID'], game['homeID']]
            rawGameDate = game['date']
            gameDate = RAPM.convertDatetimeString(rawGameDate)
            estGameDate = gameDate - timedelta(hours=4)
            if config.debug:
                print('Analyzing gameID ', game['id'], ' from ', estGameDate.strftime('%Y/%m/%d'), ' (#', gameCount,
                      ')',sep='')
            for teamID in teamIDs:
                # Get past season RAPM weighted total
                prevSeasonStr = getPrevSeasonStr(season)
                prevSeasonWeightedTeamRAPM = getPrevSeasonWeightedTeamRAPM(msf, prevSeasonStr, teamID)
                gameData.append(prevSeasonWeightedTeamRAPM)

                # Get current season RAPM weighted total
                weightedTeamRAPM = calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=True)
                gameData.append(weightedTeamRAPM)

                # Get team data and use it to calculate four factors
                seasonalStats = msf.msf_get_data(feed='seasonal_team_stats', date=estGameDate.strftime('%Y%m%d'),
                                                 team=teamID, league='nba', season=season, format='json', force='true')
                if len(seasonalStats['teamStatsTotals']) == 0:
                    fourFactorsData = [0] * 2
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(seasonalStats['teamStatsTotals'][0]['stats'])
                    basicPerGameData = extractBasicData(seasonalStats['teamStatsTotals'][0]['stats'])
                gameData.extend(fourFactorsData)
                gameData.extend(basicPerGameData)
            dateStr = estGameDate.strftime('%Y%m%d')
            gameData.extend([game['awayScore'], game['homeScore'], -100, dateStr, game['awayName'], game['awayID'],
                             game['homeName'], game['homeID']])
        except Exception as err:
            columns = getColumnNames()
            if config.debug:
                print('Error during game #', gameCount, ': ', err, sep='')
                print('\tOutputting data so far...')
            gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
            return gameDF

        gameDataArray.append(gameData)
        gameCount += 1
        time.sleep(2)

    if config.debug:
        print('Getting column names...')
    columns = getColumnNames()
    if config.debug:
        print('Outputting to dataframe...')
    gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
    return gameDF

# Given a season string, add on spread to the existing gameData frame from odds CSV file
def mergeOddsToGameData(msf, season):
    # Import the game dataset and odds CSV file
    print('Importing ', season, ' dataset and odds...', sep='')
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    oddsDF = pd.read_csv('../features/odds/' + season[:9] + '-odds.csv')
    gameDF['spread'] = -100
    numRows = oddsDF.shape[0]
    oddsIndex = 0
    # Iterate through each pair of rows in oddsDF manually
    # Build a dictionary that maps from the game (using date and both team names) to spread
    spreadDict = {}
    while oddsIndex < numRows:
        row1 = oddsDF.loc[oddsIndex, :]
        row2 = oddsDF.loc[oddsIndex + 1, :]
        dateStr = str(row1['Date'])
        if len(dateStr) == 3:
            dateStr = '0' + dateStr
        gameStr = dateStr + '/' + row1['Team'] + '/' + row2['Team']
        if row1['Close'] < row2['Close']:
            gameSpread = row1['Close']
        else:
            gameSpread = -1 * row2['Close']
        spreadDict[gameStr] = gameSpread
        oddsIndex += 2
    # Get seasonal games object
    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json',
                              force='true')
    #   Build dictionary that maps from teamID to TeamName (that matches odds CSV)
    teams = output['references']['teamReferences']
    teamDict = {}
    for team in teams:
        if team['city'] == 'Los Angeles':
            teamName = 'LA' + team['name']
        else:
            teamName = team['city'].replace(' ', '')
        teamDict[team['id']] = teamName
    # Loop through each game, find spread using spreadDict and add to gameDF
    games = output['games']
    for game in games:
        if game['schedule']['id'] not in gameDF.index:
            continue
        dateObj = RAPM.convertDatetimeString(game['schedule']['startTime']) - timedelta(hours=4)
        awayTeamID = game['schedule']['awayTeam']['id']
        homeTeamID = game['schedule']['homeTeam']['id']
        gameStr = dateObj.strftime('%m%d') + '/' + teamDict[awayTeamID] + '/' + teamDict[homeTeamID]
        gameDF.loc[game['schedule']['id'], 'spread'] = spreadDict[gameStr]

    basePath = RAPM.getBasePath(season, '', '', 'gameData')
    gameDF.to_csv(basePath + '-games.csv')

# TODO: Merge O/U to game data

def main():
    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, "MYSPORTSFEEDS")

    season = '2016-2017-regular'

    # mergeOddsToGameData(msf, season)


if __name__ == '__main__':
    main()
