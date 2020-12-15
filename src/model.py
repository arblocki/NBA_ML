
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
# from torch.backends import cudnn
import pandas as pd
import matplotlib.pyplot as plt
import random
from src.data_extract import getPrevSeasonStr
from src.RAPM import getBasePath
from src.objects.NN import NN
from src.objects.CSVDataset import CSVDataset
from src.mongo import insertGamePredictions
from math import sqrt
from os import path

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

num_inputs = 78


# determine the supported device
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')  # don't have GPU
    return device


# convert a df to tensor to be used in pytorch
def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df.values).float().to(device)


# Prepare the dataset by returning two DataLoaders
def prepare_data(dataset):
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, shuffle=False)
    test_dl = DataLoader(test, shuffle=False)
    return train_dl, test_dl


# Trains the model for one epoch of data from data_loader
# Uses optimizer to to optimize the specified criterion
def train_epoch(train_loader, model, criterion, optimizer):
    for i, (X, y) in enumerate(train_loader):
        # Clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        torch.flatten(output)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def evaluate_epoch(test_loader, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_loader):
        # inputs = inputs.to(device)
        # evaluate the model on the test set
        predicted = model(inputs)
        # retrieve numpy array
        predicted = predicted.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 2))
        # store
        predictions.append(predicted)
        actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = torch.tensor([row])
    # make prediction
    predicted = model(row)
    # retrieve numpy array
    predicted = predicted.detach().numpy()
    return predicted[0]


# Plot loss metric for each epoch of training
def plotLoss(loss, metricName='RMSE'):
    loss = loss[15:]
    rangeEnd = len(loss) + 1
    plt.plot(range(1, rangeEnd), loss)
    plt.xlabel('Epoch #')
    plt.ylabel(metricName)
    title = metricName + ' performance by Epoch'
    plt.title(title)
    plt.show()


# Given an array of objects and a row of data, output the average of the predictions of each model
def avgPredict(row, models, mses):
    predAwayScores = []
    predHomeScores = []
    for index in range(len(models)):
        model = models[index]
        mse = mses[index]
        if mse > 20:
            print('Skipping model', index, 'in prediction because RMSE is', mse)
            continue
        prediction = predict(row, model)
        predAwayScores.append(prediction[0])
        predHomeScores.append(prediction[1])
    predAwayAvg = sum(predAwayScores) / len(predAwayScores)
    predHomeAvg = sum(predHomeScores) / len(predAwayScores)
    return [predAwayAvg, predHomeAvg]


# Train and evaluate 10 different objects
def trainModelEnsemble(train_dl, test_dl, dataset):
    # Train 10 different objects at a time
    models = []
    criterions = []
    optimizers = []
    mses = []
    for i in range(10):
        # define the network
        model = NN()
        # model.to(device)
        # define the optimization
        criterion = nn.MSELoss()
        criterions.append(criterion)
        optimizer = torch.optim.Adam(model.parameters())
        optimizers.append(optimizer)

        mse = 1000
        count = 0
        while sqrt(mse) > 13.5:
            if count != 0:
                print('\t\tRMSE on model ', i, ' was ', sqrt(mse), ', getting new random split', sep='')
                train_dl, test_dl = prepare_data(dataset)
            if count == 3:
                print('\t\tReached third iteration, outputting result')
                break
            for epoch in range(20):
                # train the model
                train_epoch(train_dl, model, criterion, optimizer)
                # evaluate the model
                mse = evaluate_epoch(test_dl, model)
            count += 1

        mses.append(sqrt(mse))
        models.append(model)
        # print('\tPerformance at epoch ', epoch, ': ', sqrt(mse), sep='')
    return models, mses


# Get the string for the previous year's playoff
def getPrevPlayoffStr(season):
    if season == '2017-playoff':
        return '2016-playoff'
    elif season == '2018-playoff':
        return '2017-playoff'
    elif season == '2019-playoff':
        return '2018-playoff'
    elif season == '2020-playoff':
        return '2019-playoff'


# Given a season, simulate every day of game predictions
#   For each date of games, train on all games from previous season and this season thus far
#   Predict games using that model
#   Date parameter specifies certain date (yyyymmdd)
#   If testID is specified, write the predictions of each game to mongoDB
def simulatePicks(season, date=None, testID=None):
    if testID and not date:
        raise AttributeError('Date must be specified with testID!')

    if date:
        if len(date) != 8:
            raise ValueError('Invalid date given: ', date)
        print('Predicting games for', date)
    else:
        print('Predicting games for', season)

    prevSeasonStr = getPrevSeasonStr(season)
    twoSeasonsBackStr = getPrevSeasonStr(prevSeasonStr)
    path = '../features/gameData/' + prevSeasonStr + '-games.csv'
    twoSeasonsBackDF = pd.read_csv('../features/gameData/' + twoSeasonsBackStr + '-games.csv').set_index('gameID', drop=True)
    # load the previous season dataset
    dataset = CSVDataset(path)
    # dataset.concat(twoSeasonsBackDF)
    if season[5:] == 'playoff':
        prevPlayoffStr = getPrevPlayoffStr(season)
        prevPlayoffDF = pd.read_csv('../features/gameData/' + prevPlayoffStr + '-games.csv').set_index('gameID', drop=True)
        dataset.concat(prevPlayoffDF)

    # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    # currentSeasonDF['predAwayScore'] = -1
    # currentSeasonDF['predHomeScore'] = -1
    # currentSeasonDF['rmsError'] = -1
    currentDate = str(currentSeasonDF['date'].iloc[0])
    numRowsForDate = 0

    subDF = currentSeasonDF
    if date:
        priorGames = currentSeasonDF.loc[currentSeasonDF['date'] < int(date)]
        dataset.concat(priorGames)
        subDF = currentSeasonDF.loc[currentSeasonDF['date'] == int(date)]
        currentDate = date

    train_dl, test_dl = prepare_data(dataset)
    print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')

    models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
    rmseAvg = sum(mses) / len(mses)  # Track RMSE metric of the objects used for each pick
    print('Trained the initial model with RMSE of ', (sum(mses) / len(mses)), sep='')

    for index, row in subDF.iterrows():
        nextDate = str(row['date'])
        # If a new date is reached, add previous games onto dataset and reset model
        if nextDate != currentDate:
            # Add games from currentDate to current dataset
            # print('Adding ', numRowsForDate, ' games from ', currentDate, ' onto dataset', sep='')
            try:
                rangeEnd = currentSeasonDF.index.get_loc(index)
                rangeStart = rangeEnd - numRowsForDate
                currentDateGames = currentSeasonDF.iloc[rangeStart:rangeEnd]
                dataset.concat(currentDateGames)
                print('Outputting predictions up to', currentDate)
                basePath = getBasePath(season, '', '', 'gameData')
                currentSeasonDF.to_csv(basePath + '-games-partial.csv')
            except Exception as err:
                print('Error during gameID ', index, ': ', err, sep='')
                currentSeasonDF.to_csv('../features/gameData/until-' + currentDate + '-games.csv')
                return
            train_dl, test_dl = prepare_data(dataset)
            # print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')
            currentDate = nextDate
            numRowsForDate = 0

            models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
            rmseAvg = sum(mses) / len(mses)
            print('Retrained the model to ', nextDate, ' with RMSE of ', (sum(mses) / len(mses)), sep='')
        prediction = avgPredict(row.values[0:78].astype('float32'), models, mses)
        currentSeasonDF.loc[index, 'predAwayScore'] = prediction[0]
        currentSeasonDF.loc[index, 'predHomeScore'] = prediction[1]
        currentSeasonDF.loc[index, 'rmsError'] = rmseAvg
        numRowsForDate += 1

    if testID:
        insertGamePredictions(season, currentSeasonDF.loc[currentSeasonDF['date'] == int(date)], testID)

    return currentSeasonDF


# Given a season and threshold of when to make picks, assess the accuracy and frequency of the picks ATS
#       Optional start and end dates for custom timeframes
def assessSpreadPicks(season, threshold, dateStart=None, dateEnd=None):
    if dateStart:
        if len(dateStart) != 8:
            raise ValueError('Invalid dateStart given: ', dateStart)

        if dateEnd:
            if len(dateEnd) != 8:
                raise ValueError('Invalid dateEnd given: ', dateEnd)
            print(dateStart, '-', dateEnd, ' results with threshold = ', threshold, sep='')
        else:
            print(dateStart, ' results with threshold = ', threshold, sep='')
    else:
        print(season, ' results with threshold = ', threshold, sep='')

    unit = 10

    # Load game data into dataframe based on what dates were specified
    df = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    if dateStart and dateEnd:
        dateStartInt = int(dateStart)
        dateEndInt = int(dateEnd)
        # TODO: CHECK THAT THIS COPY WORKS
        df = df.loc[(df['date'] >= dateStartInt) & (df['date'] <= dateEndInt)]
    elif dateStart:  # If no end date specified, use Daily Games feed for startDate
        dateStartInt = int(dateStart)
        df = df.loc[df['date'] == dateStartInt]

    # Iterate through rows, make picks based on spread, and then check the actual outcome
    numPicks = 0
    numCorrect = 0
    numPushes = 0
    numGames = 0
    daySet = set()
    for index, row in df.iterrows():
        if row['predAwayScore'] == -1 or row['predHomeScore'] == -1:
            continue
        numGames += 1
        # Add date to set of unique dates
        if row['date'] not in daySet:
            daySet.add(row['date'])
        # Assess projected spread against Vegas spread
        projectedSpread = row['predAwayScore'] - row['predHomeScore']
        vegasSpread = row['spread']
        actualScoreDiff = row['awayScore'] - row['homeScore']
        if abs(projectedSpread - vegasSpread) >= threshold:
            numPicks += 1
            # Assess if pick was correct
            result = 'LOSS'
            if projectedSpread < vegasSpread and actualScoreDiff < vegasSpread:
                numCorrect += 1
                result = 'WIN'
            if projectedSpread > vegasSpread and actualScoreDiff > vegasSpread:
                numCorrect += 1
                result = 'WIN'
            if actualScoreDiff == vegasSpread:
                numPushes += 1
                result = 'PUSH'
            # print('\t\tBet on ', row['awayTeam'], '-', row['homeTeam'], ' ', row['spread'],
            #       ', pred score was ', '%.2f' % row['predAwayScore'], '-', '%.2f' % row['predHomeScore'],
            #       ', final score was ', row['awayScore'], '-', row['homeScore'], ' (', result, ')', sep='')
    numDays = len(daySet)
    print('\t%.3f picks made per day, %.3f percent of all games bet on, ' % ((numPicks / numDays), (numPicks / numGames)), numPicks, ' total', sep='')
    print('\tRecord: ', numCorrect, '-', numPicks - numCorrect - numPushes, '-', numPushes, ' (%.5f)' % (numCorrect / (numPicks - numPushes)), sep='')
    profit = (numCorrect * unit * 0.909) - ((numPicks - numCorrect - numPushes) * unit)
    percentROI = 100 * profit / (numPicks * unit)
    if profit < 0:
        print('\t', '%.2f' % profit, ' with a $', unit, ' unit per bet (', '%.2f' % percentROI, '% ROI)', sep='')
    else:
        print('\t+', '%.2f' % profit, ' with a $', unit, ' unit per bet (', '%.2f' % percentROI, '% ROI)', sep='')


# TODO:
# Given a season and expected value (EV) threshold, assess the accuracy, frequency, and profitability of ML picks
#       Optional start and end dates for custom timeframes
def assessMLPicks(season, threshold, dateStart=None, dateEnd=None):
    pass


# Given a season and threshold of when to make picks, assess the accuracy and frequency of the Over/Under picks
#       Optional start and end dates for custom timeframes
def assessOverUnderPicks(season, threshold, dateStart=None, dateEnd=None):
    # Load game data into dataframe
    df = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    # Iterate through rows, make picks based on spread, and then check the actual outcome
    numPicks = 0
    numCorrect = 0
    numPushes = 0
    numOvers = 0
    numUnders = 0
    daySet = set()
    startingIndex = 0
    for index, row in df.iloc[startingIndex:].iterrows():
        # Add date to set of unique dates
        if row['date'] not in daySet:
            daySet.add(row['date'])
        # Assess projected spread against Vegas spread
        projectedTotal = row['predAwayScore'] + row['predHomeScore']
        vegasTotal = row['overUnder']
        actualTotal = row['awayScore'] + row['homeScore']
        if abs(projectedTotal - vegasTotal) >= threshold:
            if projectedTotal < vegasTotal:
                numUnders += 1
            if projectedTotal > vegasTotal:
                numOvers += 1
            numPicks += 1
            # Assess if pick was correct
            if projectedTotal < vegasTotal and actualTotal < vegasTotal:
                numCorrect += 1
            if projectedTotal > vegasTotal and actualTotal > vegasTotal:
                numCorrect += 1
            if actualTotal == vegasTotal:
                numPushes += 1
    numGames = df.shape[0]
    numDays = len(daySet)
    print(season, ' results with threshold = ', threshold, sep='')
    print('\t%.3f picks made per day, %.3f percent of all games bet on, ' % ((numPicks / numDays), (numPicks / numGames)), numPicks, ' total', sep='')
    print('\tRecord: ', numCorrect, '-', numPicks - numCorrect - numPushes, '-', numPushes, ' (%.5f)' % (numCorrect / (numPicks - numPushes)), sep='')
    print('\t', numOvers, ' Overs, ', numUnders, ' Unders', sep='')


# Given a season and dataframe with game input data, output dataframe with predictions for each game
def predictGames(season, gameDF):
    # Concatenate the previous two seasons of data with the current season
    prevSeasonStr = getPrevSeasonStr(season)
    twoSeasonsBackStr = getPrevSeasonStr(prevSeasonStr)
    path = '../features/gameData/' + twoSeasonsBackStr + '-games.csv'
    dataset = CSVDataset(path)
    prevSeasonDF = pd.read_csv('../features/gameData/' + prevSeasonStr + '-games.csv').set_index('gameID', drop=True)
    currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID', drop=True)
    dataset.concat(prevSeasonDF)
    dataset.concat(currentSeasonDF)
    if season[5:] == 'playoff':
        prevPlayoffStr = getPrevPlayoffStr(season)
        prevPlayoffDF = pd.read_csv('../features/gameData/' + prevPlayoffStr + '-games.csv').set_index('gameID', drop=True)
        dataset.concat(prevPlayoffDF)
    train_dl, test_dl = prepare_data(dataset)
    print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')

    # Train objects
    gameDF['predAwayScore'] = -1
    gameDF['predHomeScore'] = -1
    gameDF['rmsError'] = -1
    models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
    rmseAvg = sum(mses) / len(mses)  # Track RMSE metric of the objects used for each pick
    print('Trained the initial model set with RMSE of ', (sum(mses) / len(mses)), sep='')

    # Set gameID as index and drop the column
    gameDF.set_index('gameID', drop=True)

    # Predict each game
    for index, row in gameDF.iterrows():
        # TODO: Parameterize with num_inputs
        prediction = avgPredict(row.values[1:79].astype('float32'), models, mses)
        gameDF.loc[index, 'predAwayScore'] = prediction[0]
        gameDF.loc[index, 'predHomeScore'] = prediction[1]
        gameDF.loc[index, 'rmsError'] = rmseAvg

    return gameDF


# Given a season and threshold, update the performance logs with the predicted scores and error in gameData CSV
def logPerformance(season, threshold):
    # Load game data into dataframe
    gameDF = pd.read_csv('../features/gameData/' + season + '-games.csv').set_index('gameID')
    # Iterate through rows, make picks based on spread, and then check the actual outcome
    perfDict = {}
    for index, row in gameDF.iterrows():
        indexMSE = round(row['rmsError'], 1)
        # Assess projected spread against Vegas spread
        projectedSpread = row['predAwayScore'] - row['predHomeScore']
        vegasSpread = row['spread']
        actualScoreDiff = row['awayScore'] - row['homeScore']
        if abs(projectedSpread - vegasSpread) >= threshold:
            if indexMSE not in perfDict:
                perfDict[indexMSE] = [0, 0, 0]
            # Assess if pick was correct
            if projectedSpread < vegasSpread and actualScoreDiff < vegasSpread:
                perfDict[indexMSE][0] += 1
            elif projectedSpread > vegasSpread and actualScoreDiff > vegasSpread:
                perfDict[indexMSE][0] += 1
            elif actualScoreDiff == vegasSpread:
                perfDict[indexMSE][2] += 1
            else:
                perfDict[indexMSE][1] += 1
    # Check if a performance CSV already exists
    filePath = '../features/performance/' + season + '_' + str(threshold) + '.csv'
    if path.exists(filePath):
        perfDF = pd.read_csv(filePath).set_index('rmse', drop=True)
        # Add items from this run onto the existing performance records
        for mse, record in perfDict.items():
            if mse in perfDF.index:
                perfDF.loc[mse, 'win'] += record[0]
                perfDF.loc[mse, 'loss'] += record[1]
                perfDF.loc[mse, 'push'] += record[2]
            else:
                record.append(0)
                perfDF.loc[mse] = record
        perfDF = perfDF.sort_index()
        perfDF['winPct'] = ((2 * perfDF['win']) + perfDF['push']) / (2 * (perfDF['win'] + perfDF['loss'] + perfDF['push']))
        perfDF.to_csv(filePath)
    else:
        perfDF = pd.DataFrame.from_dict(perfDict, orient='index', columns=['win', 'loss', 'push'])
        perfDF = perfDF.sort_index()  # Sort by ascending RMSE
        perfDF.index.name = 'rmse'    # Name the index column
        perfDF['winPct'] = ((2 * perfDF['win']) + perfDF['push']) / (2 * (perfDF['win'] + perfDF['loss'] + perfDF['push']))
        perfDF.to_csv(filePath)


def main():

    season = '2019-2020-regular'

    # for i in range(5):
    df = simulatePicks(season, '20200808', 'test')
    df.to_csv('../features/gameData/2019-2020-regular-games-test.csv')

    for threshold in range(0, 16):
        assessSpreadPicks(season, threshold, '20200808')
        assessSpreadPicks(season, threshold + 0.5, '20200808')
        # assessSeasonOverUnderPicks(season, threshold)
        # assessSeasonOverUnderPicks(season, threshold + 0.5)

    # logPerformance(season, 6)
    # logPerformance(season, 7)
    # logPerformance(season, 7.5)


if __name__ == '__main__':
    main()
