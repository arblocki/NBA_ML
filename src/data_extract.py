# Game Data Extraction Functions

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from ohmysportsfeedspy import MySportsFeeds
from config import config
from injuries import getInjuryDict
import RAPM

import simplejson as json
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta


# Given a timestamp in UTC ISO-6801 format, return a string (YYYYMMDD) of the previous date in EST
def getPreviousDay(timestamp):
    date = RAPM.convertDatetimeString(timestamp)
    estDate = date - timedelta(hours=4)
    dayBefore = estDate - timedelta(days=1)
    return dayBefore.strftime('%Y%m%d')


# Update the fourFactorsInputs with a game from yesterday
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


# Given a season, import the RAPM ratings from JSON file
def importPlayerRatings(season):
    path = '../features/RAPM-ratings/' + season + '-RAPM.json'
    with open(path) as inFile:
        ratings = json.load(inFile)
    ratingDict = {}
    for rating in ratings:
        ratingDict[int(rating['id'])] = rating['rating']
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
    elif season == '2017-playoff':
        return '2016-2017-regular'
    elif season == '2018-playoff':
        return '2017-2018-regular'
    elif season == '2019-playoff':
        return '2018-2019-regular'
    elif season == '2020-playoff':
        return '2019-2020-regular'
    raise ValueError('Invalid season given to getPrevSeasonStr')


# Given a team, season, and RAPM ratings, output the weightedTeamRAPM
def calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False):
    # Get MinutesPerGame for each player on the team
    output = msf.msf_get_data(feed='seasonal_player_stats', league='nba', season=season, team=teamID,
                              stats='minSecondsPerGame', format='json', force='true')
    # Map from playerID to injury status
    injuryDict = {}
    # if injuries:
    #     injuryDict = getInjuryDict()
    nameDict = {}
    if injuries:
        injuries = msf.msf_get_data(feed='player_injuries', league='nba', format='json', force='true')
        for player in injuries['players']:
            injuryDict[player['id']] = player['currentInjury']['playingProbability']
            nameDict[player['id']] = player['firstName'] + ' ' + player['lastName']
    # Record each player's minutes per game, as well as the team total (to calculate average)
    playerIDs = []
    minPerGameByPID = {}
    totalMinPerGame = 0
    for player in output['playerStatsTotals']:
        playerIDs.append(player['player']['id'])
        minutesPerGame = player['stats']['miscellaneous']['minSecondsPerGame'] / 60
        minPerGameByPID[player['player']['id']] = minutesPerGame
        totalMinPerGame += minutesPerGame
    if len(playerIDs) == 0:
        return 0
    avgMinPerGame = totalMinPerGame / len(playerIDs)
    teamTotalRAPM = 0
    # Sum every player's weightedRAPM ((minutesPerGame / teamAverageMinutesPerGame) * rating)
    numSkipped = 0
    for PID in playerIDs:
        if PID not in ratingDict:
            print('\tSkipping ', PID, ' in RAPM calculations due to lack of data', sep='')
            numSkipped += 1
            continue
        if PID in injuryDict:
            if (injuryDict[PID] == 'OUT') or (injuryDict[PID] == 'DOUBTFUL'):
                # print('\tSkipping ', nameDict[PID], ' (', PID, ') in RAPM calculations due to injury', sep='')
                print('\tSkipping ', PID, ' in RAPM calculations due to injury', sep='')
                numSkipped += 1
                continue
        rating = ratingDict[PID]
        minPerGame = minPerGameByPID[PID]
        teamTotalRAPM += (minPerGame / avgMinPerGame) * rating
    if (len(playerIDs) - numSkipped) != 0:
        weightedTeamRAPM = teamTotalRAPM / (len(playerIDs) - numSkipped)
    else:
        weightedTeamRAPM = 0

    return weightedTeamRAPM


# Given a team and season, get the weighted team RAPM
# Used for previous seasons, current seasons take a date range into account
def getPrevSeasonWeightedTeamRAPM(msf, season, teamID):
    ratingDict = importPlayerRatings(season)
    return calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False)


# Given a team and time-frame, get the weighted team RAPM
# Used for current seasons
def getTimeframeRAPM(msf, season, numStints, teamID):
    isPlayoff = (season[5:] == 'playoff')
    # Import RAPM inputs
    units, points, weights = RAPM.importPbpDataFromJSON(RAPM.getBasePath(season, '', '', 'RAPM-inputs'))
    if isPlayoff:
        regSeasonStr = getPrevSeasonStr(season)
        regSeasonUnits, regSeasonPoints, regSeasonWeights = RAPM.importPbpDataFromJSON(RAPM.getBasePath(regSeasonStr, '', '', 'RAPM-inputs'))
        regSeasonUnits.extend(units)
        regSeasonPoints.extend(points)
        regSeasonWeights.extend(weights)
        units, points, weights = regSeasonUnits, regSeasonPoints, regSeasonWeights
        numStints = len(regSeasonPoints) + numStints
    if numStints == 0:
        return 0
    ratings = RAPM.calculateRAPM(units[:numStints], points[:numStints], weights[:numStints])
    ratingDict = {}
    for rating in ratings:
        ratingDict[int(rating[0])] = rating[1]
    # Calculate RAPM from the specified window of data
    return calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False)


def combineSeasonStats(regSeasonStats, seasonalStats):
    combinedSeasonStats = {}
    combinedSeasonStats['fieldGoals'] = {}
    combinedSeasonStats['defense'] = {}
    combinedSeasonStats['freeThrows'] = {}
    combinedSeasonStats['rebounds'] = {}

    combinedSeasonStats['fieldGoals']['fgMade'] = regSeasonStats['fieldGoals']['fgMade'] + seasonalStats['fieldGoals']['fgMade']
    combinedSeasonStats['fieldGoals']['fg3PtMade'] = regSeasonStats['fieldGoals']['fg3PtMade'] + seasonalStats['fieldGoals']['fg3PtMade']
    combinedSeasonStats['fieldGoals']['fgAtt'] = regSeasonStats['fieldGoals']['fgAtt'] + seasonalStats['fieldGoals']['fgAtt']
    combinedSeasonStats['defense']['tov'] = regSeasonStats['defense']['tov'] + seasonalStats['defense']['tov']
    combinedSeasonStats['freeThrows']['ftAtt'] = regSeasonStats['freeThrows']['ftAtt'] + seasonalStats['freeThrows']['ftAtt']
    combinedSeasonStats['rebounds']['offReb'] = regSeasonStats['rebounds']['offReb'] + seasonalStats['rebounds']['offReb']
    combinedSeasonStats['rebounds']['defReb'] = regSeasonStats['rebounds']['defReb'] + seasonalStats['rebounds']['defReb']
    combinedSeasonStats['freeThrows']['ftMade'] = regSeasonStats['freeThrows']['ftMade'] + seasonalStats['freeThrows']['ftMade']

    return combinedSeasonStats


# Given a team's seasonal stats, output the Offensive and Defensive Four Factors
def calculateFourFactors(season, teamID, seasonalStats):
    teamFourFactors = []
    fourFactorsDF = pd.read_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv').set_index('id', drop=False)
    # If this is a playoff period, combine it with regular season stats
    if season[5:] == 'playoff':
        regSeasonStr = getPrevSeasonStr(season)
        regSeasonDF = pd.read_csv('../features/fourFactorsInputs/' + regSeasonStr + '-4F-inputs.csv').set_index('id', drop=False)
        fourFactorsDF.loc[teamID, 'OppFG'] += regSeasonDF.loc[teamID, 'OppFG']
        fourFactorsDF.loc[teamID, 'Opp3P'] += regSeasonDF.loc[teamID, 'Opp3P']
        fourFactorsDF.loc[teamID, 'OppFGA'] += regSeasonDF.loc[teamID, 'OppFGA']
        fourFactorsDF.loc[teamID, 'OppTOV'] += regSeasonDF.loc[teamID, 'OppTOV']
        fourFactorsDF.loc[teamID, 'OppFTA'] += regSeasonDF.loc[teamID, 'OppFTA']
        fourFactorsDF.loc[teamID, 'OppORB'] += regSeasonDF.loc[teamID, 'OppORB']
        fourFactorsDF.loc[teamID, 'OppDRB'] += regSeasonDF.loc[teamID, 'OppDRB']
        fourFactorsDF.loc[teamID, 'OppFT'] += regSeasonDF.loc[teamID, 'OppFT']

    teamOppStats = fourFactorsDF.loc[teamID]
    # Calculate effective FG%
    FG = seasonalStats['fieldGoals']['fgMade']
    threePtFG = seasonalStats['fieldGoals']['fg3PtMade']
    FGA = seasonalStats['fieldGoals']['fgAtt']
    OEFG = (FG + (0.5 * threePtFG)) / FGA
    OppFG = teamOppStats['OppFG']
    Opp3PtFG = teamOppStats['Opp3P']
    OppFGA = teamOppStats['OppFGA']
    DEFG = (OppFG + (0.5 * Opp3PtFG)) / OppFGA
    teamFourFactors.append(OEFG)
    teamFourFactors.append(DEFG)
    # Calculate turnover factors
    TOV = seasonalStats['defense']['tov']
    FTA = seasonalStats['freeThrows']['ftAtt']
    OTOV = TOV / (FGA + (0.44 * FTA) + TOV)
    OppTOV = teamOppStats['OppTOV']
    OppFTA = teamOppStats['OppFTA']
    DTOV = OppTOV / (OppFGA + (0.44 * OppFTA) + OppTOV)
    teamFourFactors.append(OTOV)
    teamFourFactors.append(DTOV)
    # Calculate rebounding factors
    OffReb = seasonalStats['rebounds']['offReb']
    OppDRB = teamOppStats['OppDRB']
    ORB = OffReb / (OffReb + OppDRB)
    DefReb = seasonalStats['rebounds']['defReb']
    OppORB = teamOppStats['OppORB']
    DRB = DefReb / (DefReb + OppORB)
    teamFourFactors.append(ORB)
    teamFourFactors.append(DRB)
    # Calculate free throw factors
    FT = seasonalStats['freeThrows']['ftMade']
    OFT = FT / FGA
    OppFT = teamOppStats['OppFT']
    DFT = OppFT / OppFGA
    teamFourFactors.append(OFT)
    teamFourFactors.append(DFT)
    return teamFourFactors


# Given a team's seasonal stats, output an array of the relevant stats only
def extractBasicData(stats):
    # 'awayGamesPlayed', 'awayFg2PtAttPerGame', 'awayFg2PtMadePerGame', 'awayFg2PtPct',
    # 'awayFg3PtAttPerGame', 'awayFg3PtMadePerGame', 'awayFg3PtPct', 'awayFgAttPerGame', 'awayFgMadePerGame',
    # 'awayFgPct', 'awayFtAttPerGame', 'awayFtMadePerGame', 'awayFtPct', 'awayOffRebPerGame', 'awayDefRebPerGame',
    # 'awayRebPerGame', 'awayAstPerGame', 'awayPtsPerGame', 'awayTovPerGame', 'awayStlPerGame', 'awayBlkPerGame',
    # 'awayBlkAgainstPerGame', 'awayPtsAgainstPerGame', 'awayFoulsPerGame', 'awayFoulsDrawnPerGame',
    # 'awayFoulPersPerGame', 'awayFoulPersDrawnPerGame', 'awayPlusMinusPerGame', 'awayWinPct'
    gamesPlayed = stats['gamesPlayed']
    basicData = [gamesPlayed,

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


# Same as above function, but adds regular season stats in
#   Have to calculate FGPct's manually
def extractBasicPlayoffData(regSeasonStats, stats):
    gamesPlayed = stats['gamesPlayed'] + regSeasonStats['gamesPlayed']
    totalWins = stats['standings']['wins'] + regSeasonStats['standings']['wins']
    totalLosses = stats['standings']['losses'] + regSeasonStats['standings']['losses']
    basicData = [gamesPlayed,

                 (stats['fieldGoals']['fg2PtAtt'] + regSeasonStats['fieldGoals']['fg2PtAtt']) / gamesPlayed,
                 (stats['fieldGoals']['fg2PtMade'] + regSeasonStats['fieldGoals']['fg2PtMade']) / gamesPlayed,
                 (stats['fieldGoals']['fg2PtMade'] + regSeasonStats['fieldGoals']['fg2PtMade']) /
                 (stats['fieldGoals']['fg2PtAtt'] + regSeasonStats['fieldGoals']['fg2PtAtt']),
                 (stats['fieldGoals']['fg3PtAtt'] + regSeasonStats['fieldGoals']['fg3PtAtt']) / gamesPlayed,
                 (stats['fieldGoals']['fg3PtMade'] + regSeasonStats['fieldGoals']['fg3PtMade']) / gamesPlayed,
                 (stats['fieldGoals']['fg3PtMade'] + regSeasonStats['fieldGoals']['fg3PtMade']) /
                 (stats['fieldGoals']['fg3PtAtt'] + regSeasonStats['fieldGoals']['fg3PtAtt']),
                 (stats['fieldGoals']['fgAtt'] + regSeasonStats['fieldGoals']['fgAtt']) / gamesPlayed,
                 (stats['fieldGoals']['fgMade'] + regSeasonStats['fieldGoals']['fgMade']) / gamesPlayed,
                 (stats['fieldGoals']['fgMade'] + regSeasonStats['fieldGoals']['fgMade']) /
                 (stats['fieldGoals']['fgAtt'] + regSeasonStats['fieldGoals']['fgAtt']),

                 (stats['freeThrows']['ftAtt'] + regSeasonStats['freeThrows']['ftAtt']) / gamesPlayed,
                 (stats['freeThrows']['ftMade'] + regSeasonStats['freeThrows']['ftMade']) / gamesPlayed,
                 (stats['freeThrows']['ftMade'] + regSeasonStats['freeThrows']['ftMade']) /
                 (stats['freeThrows']['ftAtt'] + regSeasonStats['freeThrows']['ftAtt']),

                 (stats['rebounds']['offReb'] + regSeasonStats['rebounds']['offReb']) / gamesPlayed,
                 (stats['rebounds']['defReb'] + regSeasonStats['rebounds']['defReb']) / gamesPlayed,
                 (stats['rebounds']['reb'] + regSeasonStats['rebounds']['reb']) / gamesPlayed,

                 (stats['offense']['ast'] + regSeasonStats['offense']['ast']) / gamesPlayed,
                 (stats['offense']['pts'] + regSeasonStats['offense']['pts']) / gamesPlayed,

                 (stats['defense']['tov'] + regSeasonStats['defense']['tov']) / gamesPlayed,
                 (stats['defense']['stl'] + regSeasonStats['defense']['stl']) / gamesPlayed,
                 (stats['defense']['blk'] + regSeasonStats['defense']['blk']) / gamesPlayed,
                 (stats['defense']['blkAgainst'] + regSeasonStats['defense']['blkAgainst']) / gamesPlayed,
                 (stats['defense']['ptsAgainst'] + regSeasonStats['defense']['ptsAgainst']) / gamesPlayed,

                 (stats['miscellaneous']['fouls'] + regSeasonStats['miscellaneous']['fouls']) / gamesPlayed,
                 (stats['miscellaneous']['foulsDrawn'] + regSeasonStats['miscellaneous']['foulsDrawn']) / gamesPlayed,
                 (stats['miscellaneous']['foulPers'] + regSeasonStats['miscellaneous']['foulPers']) / gamesPlayed,
                 (stats['miscellaneous']['foulPersDrawn'] + regSeasonStats['miscellaneous']['foulPersDrawn']) / gamesPlayed,
                 (stats['miscellaneous']['plusMinus'] + regSeasonStats['miscellaneous']['plusMinus']) / gamesPlayed,

                 (totalWins / (totalWins + totalLosses))]

    return basicData


# Outputs column names for CSV output
def getColumnNames():
    columns = ['gameID']
    awayColumns = ['prevawayRAPM', 'awayRAPM',  # RAPM stats
                   'awayOEFG', 'awayDEFG', 'awayOTOV', 'awayDTOV',  # Four factors calculations
                   'awayORB', 'awayDRB', 'awayOFT', 'awayDFT',
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
    columns.extend(['awayScore', 'homeScore', 'awayWin', 'homeWin', 'spread', 'overUnder', 'date',
                    'awayTeam', 'awayID', 'homeTeam', 'homeID'])
    return columns


# Given a season and timeframe, get a row of data for each game, output as Pandas dataframe
def getFinalGameData(msf, season, dateStart='', dateEnd=''):
    isPlayoff = (season[5:] == 'playoff')
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
            dateStr = estGameDate.strftime('%Y%m%d')
            prevDay = getPreviousDay(rawGameDate)
            if config.debug:
                print('Analyzing gameID ', game['id'], ' from ', estGameDate.strftime('%Y/%m/%d'), ' (#', gameCount, ')',
                      sep='')
            for teamID in teamIDs:
                # Get past season RAPM weighted total
                prevSeasonStr = getPrevSeasonStr(season)
                if isPlayoff:
                    prevSeasonStr = getPrevSeasonStr(prevSeasonStr)
                prevSeasonWeightedTeamRAPM = getPrevSeasonWeightedTeamRAPM(msf, prevSeasonStr, teamID)
                gameData.append(prevSeasonWeightedTeamRAPM)

                # Get current season RAPM weighted total
                # TODO: Rewrite this section to only calculate timeframeRAPM every calendar date, instead of every game
                if prevDay in stintDF.index:
                    numStints = stintDF.loc[prevDay, 'numStints']   # Number of stints recorded for season until day before game
                else:
                    numStints = 0
                timeframeRAPM = getTimeframeRAPM(msf, season, numStints, teamID)
                gameData.append(timeframeRAPM)

                # Get team data and use it to calculate four factors
                if prevDay in stintDF.index:
                    seasonalStats = msf.msf_get_data(feed='seasonal_team_stats', date=prevDay, team=teamID, league='nba',
                                                 season=season, format='json', force='true')
                else:
                    seasonalStats = {}
                    seasonalStats['teamStatsTotals'] = []
                if isPlayoff:
                    regSeasonStr = getPrevSeasonStr(season)
                    regSeasonStats = msf.msf_get_data(feed='seasonal_team_stats', team=teamID, league='nba',
                                                      season=regSeasonStr, format='json', force='true')
                    if len(seasonalStats['teamStatsTotals']) == 0:
                        fourFactorsData = calculateFourFactors(season, teamID, regSeasonStats['teamStatsTotals'][0]['stats'])
                        basicPerGameData = extractBasicData(regSeasonStats['teamStatsTotals'][0]['stats'])
                    else:
                        combinedSeasonStats = combineSeasonStats(regSeasonStats['teamStatsTotals'][0]['stats'],
                                                                 seasonalStats['teamStatsTotals'][0]['stats'])
                        fourFactorsData = calculateFourFactors(season, teamID, combinedSeasonStats)
                        basicPerGameData = extractBasicPlayoffData(regSeasonStats['teamStatsTotals'][0]['stats'],
                                                                   seasonalStats['teamStatsTotals'][0]['stats'])
                elif len(seasonalStats['teamStatsTotals']) == 0:
                    fourFactorsData = [0] * 8
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(season, teamID, seasonalStats['teamStatsTotals'][0]['stats'])
                    basicPerGameData = extractBasicData(seasonalStats['teamStatsTotals'][0]['stats'])
                gameData.extend(fourFactorsData)
                gameData.extend(basicPerGameData)
            boxscoreData = msf.msf_get_data(feed='game_boxscore', league='nba', season=season, game=game['id'], format='json', force='true')
            updateFourFactorsInputs(season, boxscoreData)
            awayWin = 0
            homeWin = 0
            if game['awayScore'] > game['homeScore']:
                awayWin = 1
            else:
                homeWin = 1
            gameData.extend([game['awayScore'], game['homeScore'], awayWin, homeWin, -100, -1, dateStr,
                             game['awayName'], game['awayID'], game['homeName'], game['homeID']])
        except Exception as err:
            columns = getColumnNames()
            if config.debug:
                print('Error during game #', gameCount, ': ', err, sep='')
                print('\tOutputting data so far...')
            gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
            return gameDF

        gameDataArray.append(gameData)
        gameCount += 1
        time.sleep(15)

    if config.debug:
        print('Getting column names...')
    columns = getColumnNames()
    if config.debug:
        print('Outputting to dataframe...')
    gameDF = pd.DataFrame(np.array(gameDataArray), columns=columns)
    return gameDF


# Given a date, build the input data for that day's games and return it as a dataframe
def getUpcomingGameData(msf, season, date):
    isPlayoff = (season[5:] == 'playoff')
    output = msf.msf_get_data(feed='daily_games', date=date, league='nba', season=season, format='json', force='true')
    if config.debug:
        print('\t', len(output['games']), ' upcoming games to analyze', sep='')
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

    ratingDict = {}
    if isPlayoff:
        regSeasonStr = getPrevSeasonStr(season)
        units, points, weights = RAPM.importPbpDataFromJSON('../features/RAPM-inputs/' + regSeasonStr)
        playoffUnits, playoffPoints, playoffWeights = RAPM.importPbpDataFromJSON('../features/RAPM-inputs/' + season)
        units.extend(playoffUnits)
        points.extend(playoffPoints)
        weights.extend(playoffWeights)
        ratings = RAPM.calculateRAPM(units, points, weights)
        for rating in ratings:
            ratingDict[int(rating[0])] = rating[1]
    else:
        ratingDict = importPlayerRatings(season)

    # Array of lists of game data, eventually will be exported as a dataframe
    gameDataArray = []
    gameCount = 0
    for game in games:
        # if game['id'] == 58376:
        #     continue
        try:
            gameData = [game['id']]
            teamIDs = [game['awayID'], game['homeID']]
            rawGameDate = game['date']
            gameDate = RAPM.convertDatetimeString(rawGameDate)
            estGameDate = gameDate - timedelta(hours=4)
            if config.debug:
                print('Analyzing gameID ', game['id'], ' from ', estGameDate.strftime('%Y/%m/%d'), ' (#', gameCount,
                      ')', sep='')
            for teamID in teamIDs:
                # Get past season RAPM weighted total
                prevSeasonStr = getPrevSeasonStr(season)
                if isPlayoff:
                    prevSeasonStr = getPrevSeasonStr(prevSeasonStr)
                prevSeasonWeightedTeamRAPM = getPrevSeasonWeightedTeamRAPM(msf, prevSeasonStr, teamID)
                gameData.append(prevSeasonWeightedTeamRAPM)

                # Get current season RAPM weighted total
                weightedTeamRAPM = calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=True)
                gameData.append(weightedTeamRAPM)

                # Get team data and use it to calculate four factors
                seasonalStats = msf.msf_get_data(feed='seasonal_team_stats', team=teamID, league='nba',
                                                     season=season, format='json', force='true')

                if isPlayoff:
                    regSeasonStr = getPrevSeasonStr(season)
                    regSeasonStats = msf.msf_get_data(feed='seasonal_team_stats', team=teamID, league='nba',
                                                      season=regSeasonStr, format='json', force='true')
                    if len(seasonalStats['teamStatsTotals']) == 0:
                        fourFactorsData = calculateFourFactors(season, teamID, regSeasonStats['teamStatsTotals'][0]['stats'])
                        basicPerGameData = extractBasicData(regSeasonStats['teamStatsTotals'][0]['stats'])
                    else:
                        combinedSeasonStats = combineSeasonStats(regSeasonStats['teamStatsTotals'][0]['stats'],
                                                                 seasonalStats['teamStatsTotals'][0]['stats'])
                        fourFactorsData = calculateFourFactors(season, teamID, combinedSeasonStats)
                        basicPerGameData = extractBasicPlayoffData(regSeasonStats['teamStatsTotals'][0]['stats'],
                                                                   seasonalStats['teamStatsTotals'][0]['stats'])
                elif len(seasonalStats['teamStatsTotals']) == 0:
                    fourFactorsData = [0] * 8
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(season, teamID, seasonalStats['teamStatsTotals'][0]['stats'])
                    basicPerGameData = extractBasicData(seasonalStats['teamStatsTotals'][0]['stats'])
                gameData.extend(fourFactorsData)
                gameData.extend(basicPerGameData)
            dateStr = estGameDate.strftime('%Y%m%d')
            gameData.extend([game['awayScore'], game['homeScore'], -1, -1, -100, -100, dateStr,
                             game['awayName'], game['awayID'], game['homeName'], game['homeID']])
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
def mergeSpreadToGameData(msf, season):
    # Import the game dataset and odds CSV file
    print('Importing ', season, ' dataset and odds...', sep='')
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    if season[5:] == 'playoff':
        year = int(season[0:4])
        seasonStr = str(year - 1) + '-' + str(year)
        oddsDF = pd.read_csv('../features/odds/' + seasonStr + '-odds.csv')
    else:
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


# Given a season string, add on Over/Under to the existing gameData frame from odds CSV file
def mergeOverUnderToGameData(msf, season):
    # Import the game dataset and odds CSV file
    print('Importing ', season, ' dataset and odds...', sep='')
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    if season[5:] == 'playoff':
        year = int(season[0:4])
        seasonStr = str(year - 1) + '-' + str(year)
        oddsDF = pd.read_csv('../features/odds/' + seasonStr + '-odds.csv')
    else:
        oddsDF = pd.read_csv('../features/odds/' + season[:9] + '-odds.csv')
    gameDF['overUnder'] = -100
    numRows = oddsDF.shape[0]
    oddsIndex = 0
    # Iterate through each pair of rows in oddsDF manually
    # Build a dictionary that maps from the game (using date and both team names) to spread
    overUnderDict = {}
    while oddsIndex < numRows:
        row1 = oddsDF.loc[oddsIndex, :]
        row2 = oddsDF.loc[oddsIndex + 1, :]
        dateStr = str(row1['Date'])
        if len(dateStr) == 3:
            dateStr = '0' + dateStr
        gameStr = dateStr + '/' + row1['Team'] + '/' + row2['Team']
        if row1['Close'] > row2['Close']:
            gameOverUnder = row1['Close']
        else:
            gameOverUnder = row2['Close']
        overUnderDict[gameStr] = gameOverUnder
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
        if gameStr not in overUnderDict:
            continue
        gameDF.loc[game['schedule']['id'], 'overUnder'] = overUnderDict[gameStr]

    basePath = RAPM.getBasePath(season, '', '', 'gameData')
    gameDF.to_csv(basePath + '-games.csv')


# Given a season string, calculate team four factors throughout season and add to dataset
# Output CSV with stats by team for each season (mainly relevant for current season)
def mergeFourFactorsToGameData(msf, season):
    print('Merging four factors for', season)
    # Import gameData and add columns for new factors
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    # gameDF['awayDEFG'] = 0
    # gameDF['awayOTOV'] = 0
    # gameDF['awayDTOV'] = 0
    # gameDF['awayORB'] = 0
    # gameDF['awayDRB'] = 0
    # gameDF['awayDFT'] = 0
    # gameDF['homeDEFG'] = 0
    # gameDF['homeOTOV'] = 0
    # gameDF['homeDTOV'] = 0
    # gameDF['homeORB'] = 0
    # gameDF['homeDRB'] = 0
    # gameDF['homeDFT'] = 0
    # Get list of gameIDs to go through
    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json', force='true')
    games = []
    for game in output['games']:
        games.append(game['schedule']['id'])
    # Create dataframe of stats for each team
    # teamDicts = {}
    # for team in output['references']['teamReferences']:
    #     teamDicts[team['id']] = {
    #         'id': team['id'],
    #         'abbrev': team['abbreviation'],
    #         'OppFG':  0,
    #         'Opp3P':  0,
    #         'OppFGA': 0,
    #         'OppFT':  0,
    #         'OppFTA': 0,
    #         'OppORB': 0,
    #         'OppDRB': 0,
    #         'OppTOV': 0,
    #     }
    teamDicts = pd.read_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv').set_index('id', drop=True)
    # For each game:
    gameCount = 1
    print(len(games), 'games to analyze...')
    for gameID in games:
        try:
            print('Analyzing gameID ', gameID, ' (#', gameCount, ')', sep='')
            output = msf.msf_get_data(feed='game_boxscore', league='nba', season=season, game=gameID, format='json', force='true')
            awayID = output['game']['awayTeam']['id']
            homeID = output['game']['homeTeam']['id']
            # If game is in gameData, calculate needed four factors and write to gameData
            if gameID in gameDF.index:
                prevDay = getPreviousDay(output['game']['startTime'])
                awayStatsJSON = msf.msf_get_data(feed='seasonal_team_stats', date=prevDay, team=awayID, league='nba',
                                                 season=season, format='json', force='true')
                if len(awayStatsJSON['teamStatsTotals']) != 0:
                    awayStats = awayStatsJSON['teamStatsTotals'][0]['stats']
                    gameDF.loc[gameID, 'awayDEFG'] = (
                            (teamDicts.loc[awayID, 'OppFG'] + (0.5 * teamDicts.loc[awayID, 'Opp3P'])) / teamDicts.loc[awayID, 'OppFGA']
                    )
                    gameDF.loc[gameID, 'awayOTOV'] = (
                            awayStats['defense']['tov'] / (awayStats['fieldGoals']['fgAtt'] + (0.44 * awayStats['freeThrows']['ftAtt']) + awayStats['defense']['tov'])
                    )
                    gameDF.loc[gameID, 'awayDTOV'] = (
                            teamDicts.loc[awayID, 'OppTOV'] / (teamDicts.loc[awayID, 'OppFGA'] + (0.44 * teamDicts.loc[awayID, 'OppFTA']) + teamDicts.loc[awayID, 'OppTOV'])
                    )
                    gameDF.loc[gameID, 'awayORB'] = (
                            awayStats['rebounds']['offReb'] / (awayStats['rebounds']['offReb'] + teamDicts.loc[awayID, 'OppDRB'])
                    )
                    gameDF.loc[gameID, 'awayDRB'] = (
                            awayStats['rebounds']['defReb'] / (awayStats['rebounds']['defReb'] + teamDicts.loc[awayID, 'OppORB'])
                    )
                    gameDF.loc[gameID, 'awayDFT'] = (
                            teamDicts.loc[awayID, 'OppFT'] / teamDicts.loc[awayID, 'OppFGA']
                    )
                homeStatsJSON = msf.msf_get_data(feed='seasonal_team_stats', date=prevDay, team=homeID, league='nba',
                                                     season=season, format='json', force='true')
                if len(homeStatsJSON['teamStatsTotals']) != 0:
                    homeStats = homeStatsJSON['teamStatsTotals'][0]['stats']
                    gameDF.loc[gameID, 'homeDEFG'] = (
                            (teamDicts.loc[homeID, 'OppFG'] + (0.5 * teamDicts.loc[homeID, 'Opp3P'])) / teamDicts.loc[homeID, 'OppFGA']
                    )
                    gameDF.loc[gameID, 'homeOTOV'] = (
                            homeStats['defense']['tov'] / (homeStats['fieldGoals']['fgAtt'] + (0.44 * homeStats['freeThrows']['ftAtt']) + homeStats['defense']['tov'])
                    )
                    gameDF.loc[gameID, 'homeDTOV'] = (
                            teamDicts.loc[homeID, 'OppTOV'] / (teamDicts.loc[homeID, 'OppFGA'] + (0.44 * teamDicts.loc[homeID, 'OppFTA']) + teamDicts.loc[homeID, 'OppTOV'])
                    )
                    gameDF.loc[gameID, 'homeORB'] = (
                            homeStats['rebounds']['offReb'] / (homeStats['rebounds']['offReb'] + teamDicts.loc[homeID, 'OppDRB'])
                    )
                    gameDF.loc[gameID, 'homeDRB'] = (
                            homeStats['rebounds']['defReb'] / (homeStats['rebounds']['defReb'] + teamDicts.loc[homeID, 'OppORB'])
                    )
                    gameDF.loc[gameID, 'homeDFT'] = (
                            teamDicts.loc[homeID, 'OppFT'] / teamDicts.loc[homeID, 'OppFGA']
                    )
            # Add stats from game onto fourFactors data
            awayStats = output['stats']['away']['teamStats'][0]
            homeStats = output['stats']['home']['teamStats'][0]
            #   Add homeTeamStats onto awayTeam's dict
            # awayDict = teamDicts[awayID]
            teamDicts.loc[awayID, 'OppFG'] += homeStats['fieldGoals']['fgMade']
            teamDicts.loc[awayID, 'Opp3P'] += homeStats['fieldGoals']['fg3PtMade']
            teamDicts.loc[awayID, 'OppFGA'] += homeStats['fieldGoals']['fgAtt']
            teamDicts.loc[awayID, 'OppFT'] += homeStats['freeThrows']['ftMade']
            teamDicts.loc[awayID, 'OppFTA'] += homeStats['freeThrows']['ftAtt']
            teamDicts.loc[awayID, 'OppORB'] += homeStats['rebounds']['offReb']
            teamDicts.loc[awayID, 'OppDRB'] += homeStats['rebounds']['defReb']
            teamDicts.loc[awayID, 'OppTOV'] += homeStats['defense']['tov']
            # teamDicts[awayID] = awayDict
            #   Add awayTeamStats onto homeTeam's dict
            # homeDict = teamDicts[homeID]
            teamDicts.loc[homeID, 'OppFG'] += awayStats['fieldGoals']['fgMade']
            teamDicts.loc[homeID, 'Opp3P'] += awayStats['fieldGoals']['fg3PtMade']
            teamDicts.loc[homeID, 'OppFGA'] += awayStats['fieldGoals']['fgAtt']
            teamDicts.loc[homeID, 'OppFT'] += awayStats['freeThrows']['ftMade']
            teamDicts.loc[homeID, 'OppFTA'] += awayStats['freeThrows']['ftAtt']
            teamDicts.loc[homeID, 'OppORB'] += awayStats['rebounds']['offReb']
            teamDicts.loc[homeID, 'OppDRB'] += awayStats['rebounds']['defReb']
            teamDicts.loc[homeID, 'OppTOV'] += awayStats['defense']['tov']
            # teamDicts[homeID] = homeDict

            gameCount += 1
            time.sleep(5)
        except Exception as err:
            print('Error during game #', gameCount, ': ', err, sep='')
            print('\tOutputting data so far...')
            # fourFactorDicts = teamDicts.values()
            fourFactorDF = teamDicts # pd.DataFrame(fourFactorDicts)
            gameID = games[gameCount - 1]
            finalDate = gameDF.loc[gameID, 'date']
            print('Outputting four factors DF to CSV...')
            fourFactorDF.to_csv('../features/fourFactorsInputs/' + str(finalDate) + '-4F-inputs.csv', index=True)
            print('Outputting new gameDF to CSV...')
            gameDF.to_csv('../features/gameData/' + str(finalDate) + '-games-test.csv')
            return

    # Output fourFactorDF to CSV
    # fourFactorDicts = teamDicts.values()
    fourFactorDF = teamDicts # pd.DataFrame(fourFactorDicts)
    print('Outputting four factors DF to CSV...')
    fourFactorDF.to_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv', index=True)
    print('Outputting new gameDF to CSV...')
    gameDF.to_csv('../features/gameData/' + season + '-games-test.csv')


def main():
    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, "MYSPORTSFEEDS")

    season = '2020-2021-regular'
    df = getFinalGameData(msf, season, '', '')
    df.to_csv('../features/gameData/2020-2021-regular-games-test.csv')


if __name__ == '__main__':
    main()
