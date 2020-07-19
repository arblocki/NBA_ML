
import pymongo
import pandas as pd
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


def main():
    client = pymongo.MongoClient('mongodb+srv://' + config.mongoBlock + ':' + config.mongoBlockPW +
                                 '@nba-data.nftax.azure.mongodb.net/NBA-ML?retryWrites=true&w=majority')

    insertGames(client, '2019-2020-regular')

    client.close()


if __name__ == '__main__':
    main()

