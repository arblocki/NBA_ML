# MongoDB

import pymongo
import pandas as pd
from ohmysportsfeedspy import MySportsFeeds
from src.config import config
from src.RAPM import convertDatetimeString
from datetime import timedelta


def getSeasonStr(season):
    seasonSubstr = season[:9]
    if season[5:] == 'playoff':
        year = int(season[:4])
        seasonSubstr = str(year - 1) + '-' + str(year)
    return seasonSubstr


# Given a list of teams from teamReferences, build two dicts that map from teamID to city and name
def getTeamDicts(teams):
    cityDict = {}
    nameDict = {}
    for team in teams:
        cityDict[team['id']] = team['city']
        nameDict[team['id']] = team['name']
    return cityDict, nameDict


# Given a DB instance and season string, insert all of the games into MongoDB
#   MUST BE USED WITH A SEASON THAT HAS PREDICTED SCORES
def insertGamesFromCSV(client, season):
    # Get year's string to access database
    seasonSubstr = getSeasonStr(season)
    db = client[seasonSubstr]
    coll = db['games']

    # Loop through dataframe and add each row as a dictionary to the list
    gameList = []
    df = pd.read_csv('../features/gameData/' + season + '-games.csv')
    for index, row in df.iterrows():
        # Extract data
        # TODO: Update with new schema fields
        awayTeamDict = { 'id': row['awayID'], 'abbreviation': row['awayTeam'] }
        homeTeamDict = { 'id': row['homeID'], 'abbreviation': row['homeTeam'] }
        scoreDict = { 'away': row['awayScore'], 'home': row['homeScore'] }
        predScoreDict = { 'away': row['predAwayScore'], 'home': row['predHomeScore'] }
        gameDict = { 'gameID': row['gameID'], 'date': str(row['date']), 'awayTeam': awayTeamDict, 'homeTeam': homeTeamDict,
                     'score': scoreDict, 'spread': row['spread'], 'predScore': predScoreDict }
        gameList.append(gameDict)

    # Output the list to mongo using insert_many
    result = coll.insert_many(gameList)
    print(len(result.inserted_ids), 'game objects inserted')


# Given an MSF instance, DB instance, and season, insert all unplayed games into the database
def insertUnplayedGames(msf, client, season):
    # Get year's string to access database
    seasonSubstr = getSeasonStr(season)
    db = client[seasonSubstr]
    coll = db['games']

    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='unplayed', format='json',
                              force='true')

    cityDict, nameDict = getTeamDicts(output['references']['teamReferences'])
    gameList = []
    for game in output['games']:
        # Convert UTC startTime to EST date of game
        dateObj = convertDatetimeString(game['schedule']['startTime'])
        date = (dateObj - timedelta(hours=4)).strftime('%Y%m%d')
        # Extract data
        awayTeamDict = { 'id': game['schedule']['awayTeam']['id'],
                         'abbreviation': game['schedule']['awayTeam']['abbreviation'],
                         'city': cityDict[game['schedule']['awayTeam']['id']],
                         'name': nameDict[game['schedule']['awayTeam']['id']] }
        homeTeamDict = { 'id': game['schedule']['homeTeam']['id'],
                         'abbreviation': game['schedule']['homeTeam']['abbreviation'],
                         'city': cityDict[game['schedule']['homeTeam']['id']],
                         'name': nameDict[game['schedule']['homeTeam']['id']] }
        scoreDict = { 'away': -1, 'home': -1 }
        predScoreDict = { 'away': -1, 'home': -1 }
        betDict = { 'status': '', 'team': '', 'units': 0 }
        gameDict = { 'gameID': game['schedule']['id'], 'date': str(date), 'awayTeam': awayTeamDict, 'homeTeam': homeTeamDict,
                     'score': scoreDict, 'spread': -100, 'overUnder': -100, 'predScore': predScoreDict, 'bet': betDict,
                     'startTime': game['schedule']['startTime'] }
        gameList.append(gameDict)

    result = coll.insert_many(gameList)
    print(len(result.inserted_ids), 'game objects inserted')


# Given an MSF instance, DB instance, and season string, update all games to whatever data is locally stored
def updateGames(msf, client, season, df):
    # Get year's string to access database
    seasonSubstr = getSeasonStr(season)
    db = client[seasonSubstr]
    coll = db['games']

    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, format='json', force='true')
    # Map from teamID to city and name
    cityDict, nameDict = getTeamDicts(output['references']['teamReferences'])
    # Map from gameID to gameSchedule data
    gameMap = {}
    for game in output['games']:
        gameMap[game['schedule']['id']] = game['schedule']

    # Loop through games in dataframe and update games in DB
    for index, row in df.iterrows():
        gameSchedule = gameMap[row['gameID']]
        startTime = gameSchedule['startTime']

        spreadThreshold = 6
        totalThreshold = 12
        projectedSpread = row['predAwayScore'] - row['predHomeScore']
        projectedTotal = row['predAwayScore'] + row['predHomeScore']
        vegasSpread = row['spread']
        vegasTotal = row['overUnder']
        actualScoreDiff = row['awayScore'] - row['homeScore']
        actualTotal = row['awayScore'] + row['homeScore']
        # Data on our spread bet
        units = 0
        team = ''
        status = ''
        # Data on our Over/Under bet
        totalUnits = 0
        side = ''
        totalStatus = ''
        # Check for a spread bet being made
        if abs(projectedSpread - vegasSpread) >= spreadThreshold:
            units = 1
            if projectedSpread < vegasSpread:
                team = row['homeTeam']
                status = 'LOSS'
                if actualScoreDiff < vegasSpread:
                    status = 'WIN'
            if projectedSpread > vegasSpread:
                team = row['awayTeam']
                status = 'LOSS'
                if actualScoreDiff > vegasSpread:
                    status = 'WIN'
            if actualScoreDiff == vegasSpread:
                status = 'PUSH'
        # Check for an Over/Under bet being made
        if abs(projectedTotal - vegasTotal) >= totalThreshold:
            totalUnits = 1
            if projectedTotal < vegasTotal:
                side = 'Under'
                status = 'LOSS'
                if actualTotal < vegasTotal:
                    status = 'WIN'
            if projectedTotal > vegasTotal:
                side = 'Over'
                status = 'LOSS'
                if actualTotal > vegasTotal:
                    status = 'WIN'
            if actualTotal == vegasTotal:
                status = 'PUSH'
        # Check for game with no final scores
        if row['homeScore'] == -1:
            status = ''
            totalStatus = ''
        # Prevent picks being made on unprojected games
        if row['predHomeScore'] == -1:
            units = 0
            team = ''
            totalUnits = 0
            side = ''

        query = { 'gameID': row['gameID'] }
        newValues = { '$set': {
            'startTime': startTime,
            'awayTeam.city': cityDict[gameSchedule['awayTeam']['id']],
            'awayTeam.name': nameDict[gameSchedule['awayTeam']['id']],
            'homeTeam.city': cityDict[gameSchedule['homeTeam']['id']],
            'homeTeam.name': nameDict[gameSchedule['homeTeam']['id']],
            'spread': row['spread'],
            'overUnder': row['overUnder'],
            'bet.units': units,
            'bet.team': team,
            'bet.status': status,
            'total.units': totalUnits,
            'total.side': side,
            'total.status': totalStatus,
            'predScore.away': row['predAwayScore'],
            'predScore.home': row['predHomeScore'],
        }}

        coll.update_one(query, newValues)


# Call updateGames using the dataframe from today-games.csv
def updateTodayGames(msf, client, season):
    df = pd.read_csv('../features/gameData/today-games.csv').set_index('gameID', drop=False)
    updateGames(msf, client, season, df)


# Update a game from yesterday with final score and bet results
def updateYesterdayGame(client, season, game):
    # Get year's string to access database
    seasonSubstr = getSeasonStr(season)
    db = client[seasonSubstr]
    coll = db['games']

    awayScore = game['score']['awayScoreTotal']
    homeScore = game['score']['homeScoreTotal']

    query = { 'gameID': game['schedule']['id'] }
    gameCursor = coll.find(query)
    status = ''
    totalStatus = ''
    for gameDoc in gameCursor:
        # Update bet data if a spread bet was made
        if gameDoc['bet']['units'] != 0:
            scoreDiff = awayScore - homeScore
            if gameDoc['bet']['team'] == gameDoc['awayTeam']['abbreviation']:
                if scoreDiff > gameDoc['spread']:
                    status = 'WIN'
                else:
                    status = 'LOSS'
            if gameDoc['bet']['team'] == gameDoc['homeTeam']['abbreviation']:
                if scoreDiff < gameDoc['spread']:
                    status = 'WIN'
                else:
                    status = 'LOSS'
            if scoreDiff == gameDoc['spread']:
                status = 'PUSH'
        # Update bet data is an Over/Under bet was made
        if gameDoc['total']['units'] != 0:
            total = awayScore + homeScore
            if gameDoc['total']['side'] == 'Over':
                if total > gameDoc['overUnder']:
                    totalStatus = 'WIN'
                else:
                    totalStatus = 'LOSS'
            if gameDoc['total']['side'] == 'Under':
                if total < gameDoc['overUnder']:
                    totalStatus = 'WIN'
                else:
                    totalStatus = 'LOSS'
            if total == gameDoc['overUnder']:
                totalStatus = 'PUSH'

    # Update with new values
    newValues = { '$set': {
        'score.away': awayScore,
        'score.home': homeScore,
        'bet.status': status,
        'total.status': totalStatus,
    }}

    coll.update_one(query, newValues)


def main():
    client = pymongo.MongoClient('mongodb+srv://' + config.mongoBlock + ':' + config.mongoBlockPW +
                                 '@nba-data.nftax.azure.mongodb.net/NBA-ML?retryWrites=true&w=majority')

    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, "MYSPORTSFEEDS")

    season = '2019-2020-regular'

    # df = pd.read_csv('../features/gameData/' + season + 'games.csv').set_index('gameID', drop=False)
    # updateTodayGames(msf, client, season)
    # insertUnplayedGames(msf, client, season)
    df = pd.read_csv('../features/gameData/20200806-games.csv').set_index('gameID', drop=False)
    updateGames(msf, client, season, df)

    client.close()


if __name__ == '__main__':
    main()

