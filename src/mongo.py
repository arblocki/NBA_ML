
import pymongo
import pandas as pd
from ohmysportsfeedspy import MySportsFeeds
from src.config import config


# Given a DB instance and season string, insert all of the games into MongoDB
#   MUST BE USED WITH A SEASON THAT HAS PREDICTED SCORES
def insertGames(client, season):
    # Get year's string to get database
    seasonSubstr = season[:9]
    if season[5:] == 'playoff':
        year = int(season[:4])
        seasonSubstr = str(year - 1) + '-' + str(year)
    db = client[seasonSubstr]
    coll = db['games']

    # Loop through dataframe and add each row as a dictionary to the list
    gameList = []
    df = pd.read_csv('../features/gameData/' + season + '-games.csv')
    for index, row in df.iterrows():
        # Extract data
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


# Given a DB instance and season string, update all games to whatever data is locally stored
def updateGames(msf, client, season):
    # Get year's string to get database
    seasonSubstr = season[:9]
    if season[5:] == 'playoff':
        year = int(season[:4])
        seasonSubstr = str(year - 1) + '-' + str(year)
    db = client[seasonSubstr]
    coll = db['games']

    output = msf.msf_get_data(feed='seasonal_games', league='nba', season=season, status='final', format='json',
                              force='true')
    # Map from teamID to city and name
    cityMap = {}
    nameMap = {}
    for team in output['references']['teamReferences']:
        cityMap[team['id']] = team['city']
        nameMap[team['id']] = team['name']
    # Map from gameID to gameSchedule data
    gameMap = {}
    for game in output['games']:
        gameMap[game['schedule']['id']] = game['schedule']

    df = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=False)
    # Loop through games in dataframe and update games in DB
    for index, row in df.iterrows():
        gameSchedule = gameMap[row['gameID']]
        startTime = gameSchedule['startTime']

        threshold = 6
        projectedSpread = row['predAwayScore'] - row['predHomeScore']
        vegasSpread = row['spread']
        actualScoreDiff = row['awayScore'] - row['homeScore']
        units = 0
        team = ''
        status = ''
        if abs(projectedSpread - vegasSpread) > threshold:
            units = 1
            if projectedSpread < vegasSpread:
                team = row['homeTeam']
                status = 'loss'
                if actualScoreDiff < vegasSpread:
                    status = 'win'
            if projectedSpread > vegasSpread:
                team = row['awayTeam']
                status = 'loss'
                if actualScoreDiff > vegasSpread:
                    status = 'win'
            if actualScoreDiff == vegasSpread:
                status = 'push'

        query = { 'gameID': row['gameID'] }
        newValues = { '$set': {
            'startTime': startTime,
            'awayTeam.city': cityMap[gameSchedule['awayTeam']['id']],
            'awayTeam.name': nameMap[gameSchedule['awayTeam']['id']],
            'homeTeam.city': cityMap[gameSchedule['homeTeam']['id']],
            'homeTeam.name': nameMap[gameSchedule['homeTeam']['id']],
            'bet.units': units,
            'bet.team': team,
            'bet.status': status,
            'predScore.away': row['predAwayScore'],
            'predScore.home': row['predHomeScore'],
        }}

        coll.update_one(query, newValues)


def main():
    client = pymongo.MongoClient('mongodb+srv://' + config.mongoBlock + ':' + config.mongoBlockPW +
                                 '@nba-data.nftax.azure.mongodb.net/NBA-ML?retryWrites=true&w=majority')

    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, "MYSPORTSFEEDS")

    updateGames(msf, client, '2019-2020-regular')

    client.close()


if __name__ == '__main__':
    main()

