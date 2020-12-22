import logging

import azure.functions as func
from src.model import simulatePicks


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    season = req.params.get('season')
    date = req.params.get('date')
    testID = req.params.get('testID')
    if not date:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            date = str(req_body.get('date'))
    if not testID:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            testID = str(req_body.get('testID'))
    if not season:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            season = str(req_body.get('season'))

    if season and date and testID:
        if season not in ['2017-2018-regular', '2018-playoff', '2018-2019-regular', '2019-playoff', '2019-2020-regular', '2020-playoff']:
            return func.HttpResponse(
                "Invalid season provided",
                status_code=400
            )
        if len(date) != 8:
            return func.HttpResponse(
                "Invalid date provided",
                status_code=400
            )

        logging.info(f'Starting model training and predictions for {date} ({season}) with testID {testID}.')
        simulatePicks(season, date, testID, ensemble_size=1)
        logging.info('Finished model training and predictions.')
        return func.HttpResponse(
            f"Simulated model training and predictions for {date} with testID {testID}.",
            status_code=200
        )
    else:
        return func.HttpResponse(
             "Please include a season, date, and testID with your request!",
             status_code=200
        )
