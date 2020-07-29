
import simplejson as json
import requests
from src.config import config
from ohmysportsfeedspy import MySportsFeeds

def getTodaySpreads(gameDF):
    nbaSpreads = requests.get('https://api.the-odds-api.com/v3/odds',
    params={
        'api_key': config.oddsAPI_key,
        'mkt': 'spreads',
        'region': 'us',
        'sport': 'basketball_nba',
    })
    nbaSpreadsJSON = json.loads(nbaSpreads.text)
    if not nbaSpreadsJSON['success']:
        print('There was a problem with the sports request:', nbaSpreadsJSON['msg'])

    # Import dict that maps from team abbreviation to full name
    teamDict = {}
    teamDictFilename = '../features/dictionaries/teamAbbrevToName.json'
    with open(teamDictFilename) as inFile:
        teamDict = json.load(inFile)

    # Create dictionary for each game
    spreadArray = []
    for game in nbaSpreadsJSON['data']:
        homeTeam = game['home_team']
        if game['teams'][0] == homeTeam:
            awayTeam = game['teams'][1]
            homeTeamIndex = 0
        else:
            awayTeam = game['teams'][0]
            homeTeamIndex = 1
        spread = -101
        for site in game['sites']:
            if site['site_key'] == 'bovada':
                spread = site['odds']['spreads']['points'][homeTeamIndex]
        gameDict = {
            'homeTeam': homeTeam,
            'awayTeam': awayTeam,
            'spread': spread,
        }
        spreadArray.append(gameDict)
    # Match each game in gameDF with one in our spread array then update the spread
    for index, row in gameDF.iterrows():
        for game in spreadArray:
            if game['homeTeam'] == teamDict[row['homeTeam']] and game['awayTeam'] == teamDict[row['awayTeam']]:
                gameDF.loc[index, 'spread'] = game['spread']
                break

    return gameDF


def main():
    # Create instance of MySportsFeeds API and authenticate
    msf = MySportsFeeds('2.1', verbose=False)
    msf.authenticate(config.MySportsFeeds_key, config.msfPassword)

    teamDict = {}
    output = msf.msf_get_data(feed='seasonal_team_stats', league='nba', season='2019-2020-regular', format='json', force='true')
    for team in output['teamStatsTotals']:
        teamInfo = team['team']
        teamDict[teamInfo['abbreviation']] = teamInfo['city'] + ' ' + teamInfo['name']

    unitsFilename = '../features/dictionaries/teamAbbrevToName.json'
    with open(unitsFilename, 'w') as outFile:
        json.dump(teamDict, outFile, indent=4, separators=(',', ': '))

    return 0


if __name__ == '__main__':
    main()