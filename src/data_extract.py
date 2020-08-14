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
    raise ValueError('Invalid season given to getPrevSeasonStr')


# Given a team, season, and RAPM ratings, output the weightedTeamRAPM
def calculateWeightedTeamRAPM(msf, season, teamID, ratingDict, injuries=False):
    # Get MinutesPerGame for each player on the team
    output = msf.msf_get_data(feed='seasonal_player_stats', league='nba', season=season, team=teamID,
                              stats='minSecondsPerGame', format='json', force='true')
    # Map from playerID to injury status
    injuryDict = {}
    if injuries:
        injuryDict = {
            # ATLANTA HAWKS
            9237: 'OUT',            # Clint Capela
            # BOSTON CELTICS
            # BROOKLYN NETS
            9263: 'DOUBTFUL',       # Jamal Crawford
            10155: 'OUT',           # Taurean Prince
            9209: 'OUT',            # Spencer Dinwiddie
            9268: 'OUT',            # DeAndre Jordan
            9188: 'OUT',            # Wilson Chandler
            17252: 'OUT',           # Nicholas Claxton
            9157: 'OUT',            # Kyrie Irving
            9386: 'OUT',            # Kevin Durant
            # CHARLOTTE HORNETS
            15313: 'QUESTIONABLE',  # Ray Spalding
            13815: 'OUT',           # Kobi Simmons
            # CHICAGO BULLS
            15219: 'OUT',           # Chandler Hutchison
            13862: 'OUT',           # Luke Kornet
            10129: 'OUT',           # Kris Dunn
            17276: 'OUT',           # Max Strus
            # CLEVELAND CAVALIERS
            17231: 'OUT',           # Dylan Windler
            9509: 'OUT',            # Dante Exum
            # DALLAS MAVERICKS
            10117: 'OUT',           # Dorian Finney-Smith
            9373: 'OUT',            # Porzingis
            15200: 'OUT',           # Luka Doncic
            9467: 'PROBABLE',       # Seth Curry
            9127: 'OUT',            # Courtney Lee
            9177: 'OUT',            # Dwight Powell
            15280: 'OUT',           # Jalen Brunson
            9461: 'OUT',            # Willie Cauley-Stein
            # DENVER NUGGETS
            9197: 'OUT',            # Will Barton
            9191: 'OUT',            # Gary Harris
            17236: 'OUT',           # Vlatko Cancar
            # GOLDEN STATE WARRIORS
            9227: 'OUT',            # Kevon Looney
            17331: 'OUT',           # Ky Bowman
            # HOUSTON ROCKETS
            9352: 'OUT',            # Eric Gordon
            9387: 'DOUBTFUL',       # Russ Westbrook
            9088: 'OUT',            # Thabo Sefolosha
            12199: 'OUT',           # David Nwaba
            # INDIANA PACERS
            9402: 'OUT',            # Vic Oladipo
            9252: 'OUT',            # Myles Turner
            9433: 'OUT',            # TJ Warren
            10113: 'OUT',           # Domantas Sabonis
            9131: 'OUT',            # Jeremy Lamb
            # LA CLIPPERS
            15223: 'OUT',           # Landry Shamet
            9244: 'OUT',            # Montrezl Harrell
            9239: 'OUT',            # Patrick Beverley
            # LA LAKERS
            9203: 'OUT',            # KCP
            9464: 'OUT',            # Rajon Rondo
            9097: 'OUT',            # Avery Bradley
            # MEMPHIS GRIZZLIES
            15201: 'OUT',           # Jaren Jackson Jr.
            9347: 'OUT',            # Tyus Jones
            9313: 'OUT',            # Justise Winslow
            # MIAMI HEAT
            15324: 'OUT',           # Gabe Vincent
            13742: 'OUT',           # Bam Adebayo
            9314: 'OUT',            # Goran Dragic
            9152: 'OUT',            # Jimmy Butler
            17234: 'QUESTIONABLE',  # KZ Okpala
            # MINNESOTA TIMBERWOLVES
            9346: 'OUT',            # Karl-Anthony Towns
            # NEW ORLEANS PELICANS
            16958: 'QUESTIONABLE',  # Zion Williamson
            13973: 'OUT',           # Josh Gray
            11924: 'OUT',           # Darius Miller
            # NEW YORK KNICKS
            13736: 'OUT',           # Dennis Smith Jr.
            # OKLAHOMA CITY THUNDER
            9265: 'OUT',            # Chris Paul
            9084: 'OUT',            # Dennis Schroder
            17284: 'OUT',           # Isaiah Roby
            # ORLANDO MAGIC
            9399: 'OUT',            # Evan Fournier
            9327: 'OUT',            # Michael Carter-Willaims
            9406: 'OUT',            # Aaron Gordon
            13733: 'OUT',           # Jonathan Isaac
            9452: 'OUT',            # Al-Farouq Aminu
            # PHILADELPHIA 76ERS
            9418: 'DOUBTFUL',       # Joel Embiid
            10087: 'OUT',           # Ben Simmons
            15314: 'OUT',           # Ryan Broekhoff
            15213: 'OUT',           # Zhaire Smith
            # PHOENIX SUNS
            15279: 'DOUBTFUL',      # Elie Okobo
            9211: 'OUT',            # Aron Baynes
            9526: 'OUT',            # Kelly Oubre Jr.
            17292: 'OUT',           # Tariq Owens
            # PORTLAND TRAILBLAZERS
            9312: 'QUESTIONABLE',   # Hassan Whiteside
            15312: 'PROBABLE',      # Jaylen Adams
            13752: 'OUT',           # Caleb Swanigan
            9235: 'OUT',            # Trevor Ariza
            9510: 'OUT',            # Rodney Hood
            # SACRAMENTO KINGS
            9422: 'DOUBTFUL',       # Richaun Holmes
            9086: 'QUESTIONABLE',   # Kent Bazemore
            15199: 'OUT',           # Marvin Bagley III
            # SAN ANTONIO SPURS
            13755: 'DOUBTFUL',      # Derrick White
            9099: 'OUT',            # Tyler Zeller
            10110: 'OUT',           # Bryn Forbes
            9512: 'OUT',            # Trey Lyles
            9480: 'OUT',            # LaMarcus Aldridge
            # TORONTO RAPTORS
            10112: 'OUT',           # Patrick McCaw
            17211: 'QUESTIONABLE',  # Oshae Brissett
            # UTAH JAZZ
            13741: 'QUESTIONABLE',  # Donovan Mitchell
            17320: 'PROBABLE',      # Juwan Morgan
            17269: 'OUT',           # Nigel Williams-Goss
            9114: 'OUT',            # Bojan Bogdanovic
            # WASHINGTON WIZARDS
            9405: 'QUESTIONABLE',   # Shabazz Napier
            17315: 'OUT',           # Garrison Matthews
            9523: 'OUT',            # Bradley Beal
            9522: 'OUT',            # John Wall
            10109: 'OUT',           # Davis Bertans
        }
    nameDict = {}
    # if injuries:
    #     injuries = msf.msf_get_data(feed='player_injuries', league='nba', format='json', force='true')
    #     for player in injuries['players']:
    #         injuryDict[player['id']] = player['currentInjury']['playingProbability']
    #         nameDict[player['id']] = player['firstName'] + ' ' + player['lastName']
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


# Given a team's seasonal stats, output the Offensive and Defensive Four Factors
def calculateFourFactors(season, teamID, seasonalStats):
    teamFourFactors = []
    fourFactorsDF = pd.read_csv('../features/fourFactorsInputs/' + season + '-4F-inputs.csv').set_index('id', drop=False)
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
    columns.extend(['awayScore', 'homeScore', 'spread', 'overUnder', 'date', 'awayTeam', 'awayID', 'homeTeam', 'homeID'])
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
                    fourFactorsData = [0] * 8
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(season, teamID, seasonalStats['teamStatsTotals'][0]['stats'])
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
                      ')', sep='')
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
                    fourFactorsData = [0] * 8
                    basicPerGameData = [0] * 29
                else:
                    fourFactorsData = calculateFourFactors(season, teamID, seasonalStats['teamStatsTotals'][0]['stats'])
                    basicPerGameData = extractBasicData(seasonalStats['teamStatsTotals'][0]['stats'])
                gameData.extend(fourFactorsData)
                gameData.extend(basicPerGameData)
            dateStr = estGameDate.strftime('%Y%m%d')
            gameData.extend([game['awayScore'], game['homeScore'], -100, -100, dateStr, game['awayName'], game['awayID'],
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
def mergeSpreadToGameData(msf, season):
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


# Given a season string, add on Over/Under to the existing gameData frame from odds CSV file
def mergeOverUnderToGameData(msf, season):
    # Import the game dataset and odds CSV file
    print('Importing ', season, ' dataset and odds...', sep='')
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
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
    gameDF = pd.read_csv('../features/gameData/20180201-games-test.csv').set_index('gameID', drop=True)
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
    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json', date='from-20180201-to-20180412', force='true')
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
    teamDicts = pd.read_csv('../features/fourFactorsInputs/20180201-4F-inputs.csv').set_index('id', drop=True)
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

    seasons = ['2017-2018-regular']
    for season in seasons:
        mergeFourFactorsToGameData(msf, season)


if __name__ == '__main__':
    main()
