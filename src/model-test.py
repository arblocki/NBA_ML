
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.backends import cudnn
import pandas as pd
import matplotlib.pyplot as plt
import random
from src.data_extract import getPrevSeasonStr
from src.RAPM import getBasePath
from math import sqrt
from os import path


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path).set_index('gameID', drop=True)

        # use_cuda = torch.cuda.is_available()
        # device = torch.device("cuda:0" if use_cuda else "cpu")
        # cudnn.benchmark = True

        # store the inputs and outputs
        self.X = df.values[:, 0:78].astype('float32')
        self.y = df.values[:, 78:80].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 2))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def concat(self, df):
        # store the new inputs and outputs
        newX = df.values[:, 0:78].astype('float32')
        newY = df.values[:, 78:80].astype('float32')
        # add the new data onto the existing X and y
        self.X = np.concatenate((self.X, newX), axis=0)
        self.y = np.concatenate((self.y, newY), axis=0)
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 2))

    # get indexes for train and test rows
    def get_splits(self, n_test=0.30):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


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


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(78, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 2)

        self.init_weights()

    def init_weights(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            FC_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(FC_in))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x


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


# Given an array of models and a row of data, output the average of the predictions of each model
def avgPredict(row, models):
    predAwayScores = []
    predHomeScores = []
    for model in models:
        prediction = predict(row, model)
        predAwayScores.append(prediction[0])
        predHomeScores.append(prediction[1])
    predAwayAvg = sum(predAwayScores) / len(predAwayScores)
    predHomeAvg = sum(predHomeScores) / len(predAwayScores)
    return [predAwayAvg, predHomeAvg]


# Train and evaluate 10 different models
def trainModelEnsemble(train_dl, test_dl, dataset):
    # Train 10 different models at a time
    models = []
    criterions = []
    optimizers = []
    mses = []
    for i in range(5):
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
        while (sqrt(mse) > 13.5):
            if (count != 0):
                print('\t\tRMSE on model ', i, ' was ', sqrt(mse), ', getting new random split', sep='')
                train_dl, test_dl = prepare_data(dataset)
            if (count == 3):
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


# Given a season, simulate every day of game predictions
#   Each date of games, train on all games from previous season and this season thus far
#   Predict games using that model, and make spread picks if there is a projected difference above the given threshold
def simulateSeasonPicks(season):
    prevSeasonStr = getPrevSeasonStr(season)
    twoSeasonsBackStr = getPrevSeasonStr(prevSeasonStr)
    path = '../features/gameData/' + prevSeasonStr + '-games-test.csv'
    twoSeasonsBackDF = pd.read_csv('../features/gameData/' + twoSeasonsBackStr + '-games-test.csv').set_index('gameID', drop=True)
    # load the previous season dataset
    dataset = CSVDataset(path)
    dataset.concat(twoSeasonsBackDF)
    train_dl, test_dl = prepare_data(dataset)

    # CUDA for PyTorch
    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')

    currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games-test.csv').set_index('gameID', drop=True)
    currentSeasonDF['predAwayScore'] = -1
    currentSeasonDF['predHomeScore'] = -1
    currentSeasonDF['rmsError'] = -1
    currentDate = str(currentSeasonDF['date'].iloc[0])
    numRowsForDate = 0

    models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
    rmseAvg = sum(mses) / len(mses)  # Track RMSE metric of the models used for each pick
    print('Trained the initial model with RMSE of ', (sum(mses) / len(mses)), sep='')

    for index, row in currentSeasonDF.iterrows():
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
            except Exception as err:
                print('Error during gameID ', index, ': ', err, sep='')
                currentSeasonDF.to_csv('../features/gameData/until-' + currentDate + '-games-test.csv')
                return
            train_dl, test_dl = prepare_data(dataset)
            # print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')
            currentDate = nextDate
            numRowsForDate = 0

            models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
            rmseAvg = sum(mses) / len(mses)
            print('Retrained the model to ', nextDate, ' with RMSE of ', (sum(mses) / len(mses)), sep='')
        prediction = avgPredict(row.values[0:78].astype('float32'), models)
        currentSeasonDF.loc[index, 'predAwayScore'] = prediction[0]
        currentSeasonDF.loc[index, 'predHomeScore'] = prediction[1]
        currentSeasonDF.loc[index, 'rmsError'] = rmseAvg
        numRowsForDate += 1

    basePath = getBasePath(season, '', '', 'gameData')
    currentSeasonDF.to_csv(basePath + '-games-test.csv')


# Given a season and threshold of when to make picks, assess the accuracy and frequency of the picks ATS
def assessSeasonSpreadPicks(season, threshold):
    # Load game data into dataframe
    df = pd.read_csv('../features/gameData/' + season + '-games-test.csv').set_index('gameID', drop=True)
    # Iterate through rows, make picks based on spread, and then check the actual outcome
    numPicks = 0
    numCorrect = 0
    numPushes = 0
    daySet = set()
    startingIndex = 0
    for index, row in df.iloc[startingIndex:].iterrows():
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
            if projectedSpread < vegasSpread and actualScoreDiff < vegasSpread:
                numCorrect += 1
            if projectedSpread > vegasSpread and actualScoreDiff > vegasSpread:
                numCorrect += 1
            if actualScoreDiff == vegasSpread:
                numPushes += 1
    numGames = df.shape[0]
    numDays = len(daySet)
    print(season[:9], ' results with threshold = ', threshold, sep='')
    print('\t%.3f picks made per day, %.3f percent of all games bet on, ' % ((numPicks / numDays), (numPicks / numGames)), numPicks, ' total', sep='')
    print('\tRecord: ', numCorrect, '-', numPicks - numCorrect - numPushes, '-', numPushes, ' (%.5f)' % (numCorrect / (numPicks - numPushes)), sep='')


# Given a season and threshold of when to make picks, assess the accuracy and frequency of the Over/Under picks
def assessSeasonOverUnderPicks(season, threshold):
    # Load game data into dataframe
    df = pd.read_csv('../features/gameData/' + season + '-games-test.csv').set_index('gameID', drop=True)
    # Iterate through rows, make picks based on spread, and then check the actual outcome
    numPicks = 0
    numCorrect = 0
    numPushes = 0
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
    print(season[:9], ' results with threshold = ', threshold, sep='')
    print('\t%.3f picks made per day, %.3f percent of all games bet on, ' % ((numPicks / numDays), (numPicks / numGames)), numPicks, ' total', sep='')
    print('\tRecord: ', numCorrect, '-', numPicks - numCorrect - numPushes, '-', numPushes, ' (%.5f)' % (numCorrect / (numPicks - numPushes)), sep='')


# Given a season and dataframe with game input data, output dataframe with predictions for each game
def predictGames(season, gameDF):
    # Concatenate the previous two seasons of data with the current season
    prevSeasonStr = getPrevSeasonStr(season)
    twoSeasonsBackStr = getPrevSeasonStr(prevSeasonStr)
    path = '../features/gameData/' + twoSeasonsBackStr + '-games-test.csv'
    dataset = CSVDataset(path)
    prevSeasonDF = pd.read_csv('../features/gameData/' + prevSeasonStr + '-games-test.csv').set_index('gameID', drop=True)
    currentSeasonDF = pd.read_csv('../features/gameData/' + season + '-games-test.csv').set_index('gameID', drop=True)
    dataset.concat(prevSeasonDF)
    dataset.concat(currentSeasonDF)
    train_dl, test_dl = prepare_data(dataset)
    print('Training on ', len(train_dl.dataset), ' games, Testing on ', len(test_dl.dataset), ' games', sep='')

    # Train models
    gameDF['predAwayScore'] = -1
    gameDF['predHomeScore'] = -1
    gameDF['rmsError'] = -1
    models, mses = trainModelEnsemble(train_dl, test_dl, dataset)
    rmseAvg = sum(mses) / len(mses)  # Track RMSE metric of the models used for each pick
    print('Trained the initial model set with RMSE of ', (sum(mses) / len(mses)), sep='')

    # Set gameID as index and drop the column
    gameDF.set_index('gameID', drop=True)

    # Predict each game
    for index, row in gameDF.iterrows():
        prediction = avgPredict(row.values[1:79].astype('float32'), models)
        gameDF.loc[index, 'predAwayScore'] = prediction[0]
        gameDF.loc[index, 'predHomeScore'] = prediction[1]
        gameDF.loc[index, 'rmsError'] = rmseAvg

    return gameDF


# Given a season and threshold, update the performance logs with the predicted scores and error in gameData CSV
def logPerformance(season, threshold):
    # Load game data into dataframe
    gameDF = pd.read_csv('../features/gameData/' + season + '-games-test.csv').set_index('gameID')
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
    if (path.exists(filePath)):
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
    simulateSeasonPicks(season)

    for threshold in range(3, 16):
        assessSeasonSpreadPicks(season, threshold)
        assessSeasonSpreadPicks(season, threshold + 0.5)
        # assessSeasonOverUnderPicks(season, threshold)
        # assessSeasonOverUnderPicks(season, threshold + 0.5)

    # logPerformance(season, 6)
    # logPerformance(season, 7)
    # logPerformance(season, 7.5)


if __name__ == '__main__':
    main()
