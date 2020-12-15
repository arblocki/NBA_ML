
import simplejson as json
import requests
from src.config import config
from ohmysportsfeedspy import MySportsFeeds

def getTodaySpreads(gameDF):
    # Get spreads from Bovada
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

    # Get OverUnders from Bovada
    nbaTotals = requests.get('https://api.the-odds-api.com/v3/odds',
    params={
        'api_key': config.oddsAPI_key,
        'mkt': 'totals',
        'region': 'us',
        'sport': 'basketball_nba',
    })
    nbaTotalsJSON = json.loads(nbaTotals.text)
    if not nbaTotalsJSON['success']:
        print('There was a problem with the sports request:', nbaTotalsJSON['msg'])

    # Import dict that maps from team abbreviation to full name
    teamDictFilename = '../features/dictionaries/teamAbbrevToName.json'
    with open(teamDictFilename) as inFile:
        teamDict = json.load(inFile)

    # Create dictionary for each game
    oddsArray = []
    for game in nbaSpreadsJSON['data']:
        homeTeam = game['home_team']
        if game['teams'][0] == homeTeam:
            awayTeam = game['teams'][1]
            homeTeamIndex = 0
        else:
            awayTeam = game['teams'][0]
            homeTeamIndex = 1
        spread = -100
        total = -100
        oddsSite = 'bovada'
        for site in game['sites']:
            if site['site_key'] == oddsSite:
                spread = site['odds']['spreads']['points'][homeTeamIndex]
        # Find total for this game
        for gameTotal in nbaTotalsJSON['data']:
            if gameTotal['teams'][homeTeamIndex] == homeTeam:
                for siteTotal in gameTotal['sites']:
                    if siteTotal['site_key'] == oddsSite:
                        total = siteTotal['odds']['totals']['points'][0]

        gameDict = {
            'homeTeam': homeTeam,
            'awayTeam': awayTeam,
            'spread': spread,
            'overUnder': total,
        }
        if spread != -100 and total != -100:
            oddsArray.append(gameDict)

    # Match each game in gameDF with one in our spread array then update the spread and Over/Under
    for index, row in gameDF.iterrows():
        for game in oddsArray:
            if game['homeTeam'] == teamDict[row['homeTeam']] and game['awayTeam'] == teamDict[row['awayTeam']]:
                print('Setting ', row['awayTeam'], '-', row['homeTeam'], ' to ', game['spread'], sep='')
                print('\tOverUnder at ', game['overUnder'])
                gameDF.loc[index, 'spread'] = game['spread']
                gameDF.loc[index, 'overUnder'] = game['overUnder']
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
