# RAPM Calculation

from ohmysportsfeedspy import MySportsFeeds
from src.config import config

import simplejson as json
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model


# Given a string timestamp in UTC ISO-6801, return a datetime object
def convertDatetimeString(dateString):
    return datetime.strptime(dateString, '%Y-%m-%dT%H:%M:%S.000Z')


# Given a play, update possessions or points for the correct
def updateStintStats(play, homePoss, awayPoss, homePts, awayPts, homeTeamID):
    # If jump ball, add a possession to the winning team
    if 'jumpBall' in play:
        if play['jumpBall']['wonBy'] == 'AWAY':
            awayPoss += 1
        else:
            homePoss += 1
    # On defensive rebound, or turnover, adjust possessions accordingly
    elif ('rebound' in play) and (play['rebound']['type'] == "DEFENSIVE"):
        if play['rebound']['team']['id'] == homeTeamID:
            homePoss += 1
        else:
            awayPoss += 1
    elif 'turnover' in play:  # Offensive fouls are registered as fouls AND turnovers, so doing both would double count offensive foul turnovers
        if play['turnover']['team']['id'] == homeTeamID:
            awayPoss += 1
        else:
            homePoss += 1
    # On FG or FT, adjust points accordingly
    elif ('fieldGoalAttempt' in play) and (play['fieldGoalAttempt']['result'] == "SCORED"):
        if play['fieldGoalAttempt']['team']['id'] == homeTeamID:
            homePts += play['fieldGoalAttempt']['points']
            awayPoss += 1
        else:
            awayPts += play['fieldGoalAttempt']['points']
            homePoss += 1
    elif ('freeThrowAttempt' in play) and (play['freeThrowAttempt']['result'] == "SCORED"):
        if play['freeThrowAttempt']['team']['id'] == homeTeamID:
            homePts += 1
            if play['freeThrowAttempt']['attemptNum'] == play['freeThrowAttempt']['totalAttempts']:
                awayPoss += 1
        else:
            awayPts += 1
            if play['freeThrowAttempt']['attemptNum'] == play['freeThrowAttempt']['totalAttempts']:
                homePoss += 1

    return homePoss, awayPoss, homePts, awayPts


# Given a play, update possessions for the correct team
def updatePossessions(play, homePoss, awayPoss, homeTeamID):
    # If jump ball, add a possession to the winning team
    if 'jumpBall' in play:
        if play['jumpBall']['wonBy'] == 'AWAY':
            awayPoss += 1
        else:
            homePoss += 1
    # On defensive rebound, or turnover, adjust possessions accordingly
    elif ('rebound' in play) and (play['rebound']['type'] == "DEFENSIVE"):
        if play['rebound']['team']['id'] == homeTeamID:
            homePoss += 1
        else:
            awayPoss += 1
    elif 'turnover' in play:  # Offensive fouls are registered as fouls AND turnovers, so doing both would double count offensive foul turnovers
        if play['turnover']['team']['id'] == homeTeamID:
            awayPoss += 1
        else:
            homePoss += 1
    # On FG or FT, adjust points accordingly
    elif ('fieldGoalAttempt' in play) and (play['fieldGoalAttempt']['result'] == "SCORED"):
        if play['fieldGoalAttempt']['team']['id'] == homeTeamID:
            awayPoss += 1
        else:
            homePoss += 1
    elif ('freeThrowAttempt' in play) and (play['freeThrowAttempt']['result'] == "SCORED"):
        if play['freeThrowAttempt']['team']['id'] == homeTeamID:
            if play['freeThrowAttempt']['attemptNum'] == play['freeThrowAttempt']['totalAttempts']:
                awayPoss += 1
        else:
            if play['freeThrowAttempt']['attemptNum'] == play['freeThrowAttempt']['totalAttempts']:
                homePoss += 1

    return homePoss, awayPoss


# Get play-by-play data from a data range and output units, points, and weights
#       Get all play-by-play data for games played between [dateStart, dateEnd), then
#       output the 3 lists of data that represent each stint
#       * 'units' is a list of dicts of the home and away players for each stint with a +1/-1 indicating
#           which team they're on (home/away)
#       * 'points' is a list of the point differentials (normalized to 100 possessions) for each stint
#       * 'weights' is a list of the number of possessions for each stint
def extractPbpData(msf, season, dateStart='', dateEnd=''):
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

    games = []
    for game in output['games']:
        games.append(game['schedule']['id'])

    # config.debug
    # print(games)
    print("Analyzing ", len(games), " games of data", sep='')

    units = []
    points = []
    weights = []
    numGames = 0
    # For each game:
    for gameID in games:
        try:
            print("Analyzing gameID ", gameID, " (", numGames, ")", sep='')
            numGames += 1

            # Use Game Lineup feed to get starting lineup for first stint
            try:
                output = msf.msf_get_data(game=gameID, feed='game_lineup', league='nba', season=season, format='json',
                                          force='true')
            except Warning as apiError:
                print("Warning received on game ", gameID, " lineup", sep='')
                print('\t', apiError, sep='')
                continue

            try:
                away = output['teamLineups'][0]['actual']['lineupPositions']
            except TypeError:
                print("No actual lineup provided, using expected starting lineup")
                # print(json.dumps(output, indent=4, separators=(',', ': ')))
                away = output['teamLineups'][0]['expected']['lineupPositions']
            awayUnit = []
            for awayPlayer in away:
                if 'Starter' in awayPlayer['position']:
                    awayUnit.append(awayPlayer['player']['id'])

            try:
                home = output['teamLineups'][1]['actual']['lineupPositions']
            except TypeError:
                print("No actual lineup provided, using expected starting lineup")
                # print(json.dumps(output, indent=4, separators=(',', ': ')))
                home = output['teamLineups'][1]['expected']['lineupPositions']
            homeUnit = []
            for homePlayer in home:
                if 'Starter' in homePlayer['position']:
                    homeUnit.append(homePlayer['player']['id'])
            homeTeamID = output['game']['homeTeam']['id']

            # For each stint, add up points and possessions for each side
            # Output each stint to units, points, and weights
            try:
                output = msf.msf_get_data(game=gameID, feed='game_playbyplay', league='nba', season=season,
                                          format='json', force='true')
            except Warning as apiError:
                print("Warning received on game ", gameID, " play-by-play", sep='')
                print('\t', apiError, sep='')
                continue
            plays = output['plays']
            numPlays = len(plays)
            playIndex = 0
            homePoss = 0
            awayPoss = 0
            homePts = 0
            awayPts = 0
            while playIndex < numPlays:
                # print("\tplay", playIndex, sep='')
                play = plays[playIndex]

                # On substitution, output stint data if there is enough data
                if 'substitution' in play:

                    # Output stint data to units, points, and weights IF possessions >= 1
                    if (homePoss + awayPoss) >= 2:
                        # Calculate point differential (normalized to 100 possessions)
                        pointDiff = 100 * (homePts - awayPts) / ((homePoss + awayPoss + 0.01) / 2.)
                        # Turn home and away units into dictionaries
                        homeUnitDict = {homeId: 1 for homeId in homeUnit}
                        awayUnitDict = {awayId: -1 for awayId in awayUnit}
                        stint = homeUnitDict.copy()
                        stint.update(awayUnitDict)
                        # Record stint data
                        units.append(stint)
                        points.append(pointDiff)
                        weights.append((homePoss + awayPoss) / 2.)

                        # config.debug
                        # print("after stint #", len(units), ", home team has ", homePts, " points on ", homePoss, " possessions", sep='')
                        # print("                away team has ", awayPts, " points on ", awayPoss, " possessions", sep='')

                        homePts = 0
                        awayPts = 0
                        homePoss = 0
                        awayPoss = 0

                    # Update away/home unit
                    if play['substitution']['outgoingPlayer'] is None:
                        playerOut = -1
                    else:
                        playerOut = play['substitution']['outgoingPlayer']['id']

                    if play['substitution']['incomingPlayer'] is None:
                        playerIn = -1
                        # print("Null player being subbed in on play ", playIndex, sep='')
                    else:
                        playerIn = play['substitution']['incomingPlayer']['id']
                    if play['substitution']['team']['id'] == homeTeamID:
                        if playerOut != -1 and playerIn != -1:
                            homeUnit[:] = [playerIn if homePlayer == playerOut else homePlayer for homePlayer in
                                           homeUnit]
                        elif playerOut == -1 and playerIn != -1:
                            if playerIn not in homeUnit:
                                homeUnit.append(playerIn)
                        elif playerOut != -1 and playerIn == -1:
                            try:
                                homeUnit.remove(playerOut)
                            except ValueError:
                                print("\tTried to remove ID ", playerOut, " from the homeUnit ", homeUnit, sep='')
                    else:
                        if playerOut != -1 and playerIn != -1:
                            awayUnit[:] = [playerIn if awayPlayer == playerOut else awayPlayer for awayPlayer in
                                           awayUnit]
                        elif playerOut == -1 and playerIn != -1:
                            if playerIn not in homeUnit:
                                awayUnit.append(playerIn)
                        elif playerOut != -1 and playerIn == -1:
                            try:
                                awayUnit.remove(playerOut)
                            except ValueError:
                                print("\tTried to remove ID ", playerOut, " from the awayUnit ", awayUnit, sep='')

                homePoss, awayPoss, homePts, awayPts = updateStintStats(play, homePoss, awayPoss, homePts, awayPts,
                                                                        homeTeamID)
                playIndex += 1
            time.sleep(3)
        except Exception as err:
            print('Error during game ', gameID, ': ', err, sep='')
            continue

    # Return lists
    return units, points, weights


# Given a time range, calculate the total number of stints played during that time
def getStintNumber(msf, season, dateStart='', dateEnd=''):
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

    games = []
    for game in output['games']:
        games.append(game['schedule']['id'])

    if config.debug:
        print("\tAnalyzing ", len(games), " games of data", sep='')

    numStints = 0
    numGames = 0
    # For each game:
    for gameID in games:
        try:
            if config.debug:
                print("Analyzing gameID ", gameID, " (", numGames, ")", sep='')
            numGames += 1

            try:
                output = msf.msf_get_data(game=gameID, feed='game_playbyplay', league='nba', season=season,
                                          format='json', force='true')
            except Warning as apiError:
                print("Warning received on game ", gameID, " play-by-play", sep='')
                print('\t', apiError, sep='')
                continue
            plays = output['plays']
            numPlays = len(plays)
            playIndex = 0
            homeTeamID = output['game']['homeTeam']['id']
            homePoss = 0
            awayPoss = 0
            while playIndex < numPlays:
                # print('\tplay', playIndex, sep='')
                play = plays[playIndex]

                # On substitution, output stint data if there is enough data
                if 'substitution' in play:

                    # Output stint data to units, points, and weights IF possessions >= 1
                    if (homePoss + awayPoss) >= 2:
                        numStints += 1

                        homePoss = 0
                        awayPoss = 0

                homePoss, awayPoss = updatePossessions(play, homePoss, awayPoss, homeTeamID)
                playIndex += 1
            time.sleep(3)

        except Exception as err:
            print('Error during game ', gameID, ': ', err, sep='')
            continue

    if config.debug:
        print('\tFound ', numStints, ' total stints', sep='')

    return numStints


# Given a season, output the number of stints played on each day to CSV
def outputStintCSV(msf, season):
    # Get dates between which there are final games
    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json',
                              force='true')
    games = output['games']
    startDate = convertDatetimeString(games[0]['schedule']['startTime'])
    endDate = convertDatetimeString(games[len(games) - 1]['schedule']['startTime'])
    stintArray = []
    # Starting from the first date:
    numStints = 0
    while (startDate - timedelta(days=1)).strftime('%Y-%m-%d') != endDate.strftime('%Y-%m-%d'):
        # Get number of stints and add it to the array
        if config.debug:
            print('Getting stints for ', startDate.strftime('%Y-%m-%d'), '...', sep='')
        try:
            numStints += getStintNumber(msf, season, dateStart=startDate.strftime('%Y%m%d'))
        except Exception as err:
            # If there's an error, export what we have so far
            print('Error received on ', startDate.strftime('%Y%m%d'), ': ', err, sep='')
            print('\tExporting what we have, up to', (startDate - timedelta(days=1)).strftime('%Y%m%d'))
            df = pd.DataFrame(np.array(stintArray), columns=['date', 'numStints'])
            df.to_csv('../features/stintsByDate/' + season + '-stints.csv')
        stintArray.append([startDate.strftime('%Y%m%d'), numStints])
        if config.debug:
            print('\t', numStints, ' found through ', startDate.strftime('%Y-%m-%d'), sep='')
        # Increment by 1 day using timedelta
        startDate = startDate + timedelta(days=1)

    df = pd.DataFrame(np.array(stintArray), columns=['date', 'numStints'])
    df.to_csv('../features/stintsByDate/' + season + '-stints.csv')


# Given units, points, and weights values, calculates RAPM for each player and outputs a list of pairs
# as (PlayerID, RAPM)
def calculateRAPM(units, points, weights):
    u = DictVectorizer(sparse=False)
    u_mat = u.fit_transform(units)

    # config.debug
    # print(u_mat)
    # print(points[:25])
    # print(weights[:100])

    playerIDs = u.get_feature_names()
    # print(json.dumps(u.get_feature_names()[:25], indent=4*' '))
    # print(json.dumps(u.inverse_transform(u_mat)[:1], indent=4 * ' '))

    clf = linear_model.RidgeCV(alphas=(np.array([0.01, 0.1, 1.0, 10, 100, 500, 1000, 2000, 5000])), cv=5)
    clf.fit(u_mat, points, sample_weight=weights)
    # print(clf.alpha_)
    ratings = []
    for playerID in playerIDs:
        ratings.append((playerID, clf.coef_[playerIDs.index(playerID)]))
    ratings.sort(key=lambda tup: tup[1], reverse=True)

    return ratings


# Given a season (and API instance), outputs a dictionary that maps playerID to full name
def getPlayerNames(msf, season):
    output = msf.msf_get_data(feed='players', league='nba', season=season, format='json', force='true')
    playerDict = {}  # Maps playerID to full name
    for player in output['players']:
        playerDict[player['player']['id']] = player['player']['firstName'] + ' ' + player['player']['lastName']

    return playerDict


# Given strings of season, start date, and end date, get the path to the saved files of features
def getBasePath(season, dateStart, dateEnd, dataType):
    basePath = '../features/' + dataType + '/'
    if dateStart == '':
        basePath += season
    elif dateEnd == '':
        basePath += dateStart
    else:
        basePath += (dateStart + '-to-' + dateEnd)
    return basePath


# Given units, points, and weights values as well as the season, outputs each of them to a CSV file
def exportPbpDataToJSON(units, points, weights, basePath):
    # Export units
    unitsFilename = basePath + '-units.json'
    with open(unitsFilename, 'w') as outFile:
        json.dump(units, outFile, indent=4, separators=(',', ': '))
    # Export points
    pointsFilename = basePath + '-points.json'
    with open(pointsFilename, 'w') as outFile:
        json.dump(points, outFile, indent=4, separators=(',', ': '))
    # Export weights
    weightsFilename = basePath + '-weights.json'
    with open(weightsFilename, 'w') as outFile:
        json.dump(weights, outFile, indent=4, separators=(',', ': '))


# Given a season as input, import
def importPbpDataFromJSON(basePath):
    # Import units from CSV
    unitsFilename = basePath + '-units.json'
    with open(unitsFilename) as inFile:
        units = json.load(inFile)
    # Import points from CSV
    pointsFilename = basePath + '-points.json'
    with open(pointsFilename) as inFile:
        points = json.load(inFile)
    # Import weights from CSV
    weightsFilename = basePath + '-weights.json'
    with open(weightsFilename) as inFile:
        weights = json.load(inFile)

    return units, points, weights


# Output RAPM ratings
def exportPlayerRatings(ratings, playerDict, basePath):
    # Create new object with playerID, name, and RAPM
    ratingsWithName = []
    for rating in ratings:
        nextPlayer = {'id': rating[0], 'name': playerDict[rating[0]], 'rating': rating[1]}
        ratingsWithName.append(nextPlayer)
    # Output ratings object
    filename = basePath + '-RAPM.json'
    with open(filename, 'w') as outFile:
        json.dump(ratingsWithName, outFile, indent=4, separators=(',', ': '))


def main():
    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, "MYSPORTSFEEDS")

    # seasons = ['2016-playoff', '2017-playoff', '2018-playoff', '2019-playoff']
    season = '2016-2017-regular'
    outputStintCSV(msf, season)
    # dateStart = ''
    # dateEnd = ''
    # printRatings = False

    # # for season in seasons:
    # ratingBasePath = getBasePath(season, dateStart, dateEnd, 'RAPM-ratings')
    # inputBasePath = getBasePath(season, dateStart, dateEnd, 'RAPM-inputs')
    #
    # if config.debug: print("Analyzing play-by-play data for " + season + "... ")
    # units, points, weights = extractPbpData(msf, season, dateStart, dateEnd)
    # # unitsImported, pointsImported, weightsImported = importPbpDataFromJSON(inputBasePath)
    #
    # if config.debug: print("Getting player names...")
    # playerDict = getPlayerNames(msf, season)
    #
    # if config.debug: print("exporting play-by-play to JSON... ")
    # exportPbpDataToJSON(units, points, weights, inputBasePath)
    #
    # if config.debug: print("Calculating RAPM...")
    # ratings = calculateRAPM(units, points, weights)
    # # ratingsImported = calculateRAPM(unitsImported, pointsImported, weightsImported)
    #
    # if printRatings:
    #     for rating in ratings:
    #         print(rating[0], "{}".format(playerDict[rating[0]]), "{0:.3f}".format(rating[1]))
    #
    # if config.debug: print("Export RAPM ratings...")
    # exportPlayerRatings(ratings, playerDict, ratingBasePath)


if __name__ == '__main__':
    main()
