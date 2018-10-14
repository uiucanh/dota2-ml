import logging
import dota2api
import requests
from pymongo import MongoClient, errors
from dota2api import exceptions
from src.constants import *
from datetime import datetime

def initialise_keys():
    # Initialise api keys
    if D2API_KEY is None:
        raise NameError("Dota 2 Api key needs to be set as an environment variable")
    if OD_KEY is None:
        raise NameError("Opendota Api key needs to be set as an environment variable")
    api = dota2api.Initialise(D2API_KEY, raw_mode = True)           #raw mode to get hero id instead of names
    return api

def setup_db_log(log_name):
    # Initialise logger
    logging.basicConfig(filename=log_name, level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.ERROR)       #Add this to stop requests from logging
    logger = logging.getLogger(__name__)

    #Initialise database
    client = MongoClient(SERVER_ADDRESS)
    db_collection = client.dota2db.matches
    #Test connection
    try:
        client.server_info()
    except errors.ServerSelectionTimeoutError:
        error_text = "Could not connect to database"
        logger.error(error_text)
        exit()

    return db_collection, logger

'''
- Get randomly public matches from Opendota
- Validate these matches
'''

def get_rand_public_games(use_key = False, initial_match_id = None):
    url = 'https://api.opendota.com/api/publicMatches'
    if initial_match_id != None:
        url += '?less_than_match_id=' + str(initial_match_id)
    if use_key == True:
        if initial_match_id != None:
            url += '&' + OD_KEY
        else:
            url += '?' + OD_KEY
    try:
        result = requests.get(url).json()
    except requests.exceptions.RequestException as error:
        error_text = "Could not call the Opendota API. Error Code: %s" % error
        logger.error(error_text)
        exit()
    else:
        return result

#Return True if a player abandoned
def check_for_abandons(match):
    try:
        result = api.get_match_details(match_id = match)
    except exceptions.APIError as error:
        logger.error("Could not call the Steam API. Error Code: %s" % error)
        exit()
    else:
        for player in result['players']:
            if player['leaver_status'] not in NON_ABANDON_VALUES:
                return True
        return False

'''
Validate matches, ensuring that they match the following criterias:
- Must be ranked All Pick
- Must be at least 15 mins long
- Must be at least Acient (avg medal = 60)
- No one abandoned
'''

def validate_match(match_list, len_current_list = 0):
    check_list = []
    passed_list = []
    #Check if the game mode, rank tier, lobby type and duration are correct
    for match in match_list:
        rank_tier = match['avg_rank_tier']
        match_duration = match['duration'] / 60     #In minutes
        lobby_type = match['lobby_type']
        game_mode = match['game_mode']
        #Check if all rules met
        rules = [rank_tier >= MIN_RANK_TIER,
                 match_duration >= MIN_MATCH_DURATION,
                 lobby_type == RANKED_LOBBY_MODE,               #Ranked
                 game_mode in GAME_MODE]                        #All pick and Captain Mode
        if all(rules):
            check_list.append(match)

        if len(check_list) + len_current_list == API_CALL_LIMIT:
            break

    #Check to not include matches that have abandoned players
    for match in check_list:
        match_id = match['match_id']

        if not check_for_abandons(match_id):
            passed_list.append(match)
    return passed_list

def add_json_to_db(match_list):
    #Clean up jsons
    success = 0            #Number of success cases
    keys_to_delete = ['match_seq_num', 'num_mmr', 'lobby_type',
                      'num_rank_tier', 'cluster']
    for match in match_list:
        for key in keys_to_delete:
            match.pop(key)

    #Add to db
    for match in match_list:
        try:
            db_collection.insert_one(match)
        except errors.WriteError:
            continue
        else:
            success += 1

    # If cant find any more matches
    if success == 0:
        logger.info("No more matches to collect")
        exit()
    logger.info("Number of success cases: %s" % success)

# An aggregate pipeline to check if any match is duplicated in the database
def check_for_duplicates():
    pipeline = [
        {"$group": {"_id": {"match_id": "$match_id"}, "uniqueIds": {"$addToSet": "$_id"}, "count": {"$sum": 1}}},
        {"$match": {"count": {"$gt": 1}}}
    ]
    check = print(list(db_collection.aggregate(pipeline)))
    if check != []:
        return False
    return True

def main():
    last_match_id = 4042096908
    passed_matches = []
    while True:
        rand_match = get_rand_public_games(initial_match_id = last_match_id)
        #Set new search to include matches with lower id only
        last_match_id = rand_match[-1]['match_id']

        if len(rand_match) == 0:
            rand_match = get_rand_public_games(initial_match_id = last_match_id, use_key = True)       #Use key if reach limit

        passed_matches += validate_match(rand_match, len(passed_matches))
        print(len(passed_matches))

        if len(passed_matches) >= API_CALL_LIMIT:
            add_json_to_db(passed_matches)
            total_matches = db_collection.count()
            logger.info('Last match ID: %s' % passed_matches[-1]['match_id'])
            start_time = datetime.fromtimestamp(passed_matches[-1]['start_time']).strftime('%Y-%m-%d %H:%M:%S')
            logger.info('Last match start time: %s' % start_time)
            logger.info('Finish processing %s matches\nTotal matches: %s\nContinuing' % (API_CALL_LIMIT, total_matches))
            passed_matches = []

if __name__ == '__main__':
    api = initialise_keys()
    db_collection, logger = setup_db_log('mining.log')
    main()