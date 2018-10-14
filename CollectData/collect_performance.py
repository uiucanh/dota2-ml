import requests
import numpy as np

import multiprocessing
from collect_data import setup_db_log, initialise_keys

api = initialise_keys()
db_collection, logger = setup_db_log('collect_performance.log')
session = requests.Session()


# Return a dictionary in the format Hero_id : Account_id from a match
def get_acc_list(match_id):
    match_details = api.get_match_details(match_id=match_id)
    hero_acc_dict = {}

    for player in range(len(match_details['players'])):
        acc_id = match_details['players'][player]['account_id']
        hero_id = match_details['players'][player]['hero_id']
        hero_acc_dict[hero_id] = acc_id
    return hero_acc_dict


# Go through the match using the Hero: Account dictionary
# Find the number of Ranked games each player has on each hero
# If the total is less than 10 or if they set their profile private, i.e. len(result) = 0
# set the performance to be 0.5
# else find the win percentage and add to the list of performance for each team
def get_players_urls(hero_acc_dict, match_id):
    urls = []

    for hero_id in hero_acc_dict:
        url = 'https://api.stratz.com/api/v1/Player/{}/heroPerformance/{}?lobbyType=7'.format(hero_acc_dict[hero_id],
                                                                                              hero_id)
        urls.append(url)
    if len(urls) != 10:
        error_text = 'Error in getting player urls from %s. Not enough length' % match_id
        logger.error(error_text)
        raise Exception(error_text)
    return urls


def get_players_performance(urls, match_id):
    # Get the players win-loss count
    results = []
    rad_performance, dire_performance = [], []
    num_privates = 0
    radiant_win = db_collection.find_one({'match_id': match_id})['radiant_win']

    for i in range(len(urls)):
        try:
            result = session.get(urls[i])
        except requests.exceptions.RequestException as error:
            error_text = "Could not call the Stratz API. Error Code: %s" % error
            logger.error(error_text)
            exit()
        else:
            # If player profile is private
            if result.text == '':
                # Set the avg performance to be 0.5 (10/20)
                results.append({'total': 20, 'win': 10})
                num_privates += 1
            else:
                # Convert to json
                result = result.json()
                total_games = result['matchCount'] - 1      # excluding the current match
                win_count = result['winCount']
                results.append({'total': total_games, 'win': win_count})

    # Iterating through the radiant team
    for i in range(len(results) - 5):
        if results[i]['total'] < 10:
            rad_performance.append(0.5)
        else:
            if radiant_win:
                win_percentage = (results[i]['win'] -1) / results[i]['total']
            else:
                win_percentage = results[i]['win']  / results[i]['total']
            rad_performance.append(win_percentage)

    # Iterating through the Dire team
    for i in range(5, len(results)):
        if results[i]['total'] < 10:
            dire_performance.append(0.5)
        else:
            if radiant_win:
                win_percentage = results[i]['win'] / results[i]['total']
            else:
                win_percentage =   (results[i]['win'] -1) / results[i]['total']
            dire_performance.append(win_percentage)

    if num_privates == 10:
        logger.error('10 num privates at match id: %s' % match_id)
        exit()
    logger.info('Finished processing for match id: %s' % match_id)
    return np.mean(rad_performance), -np.mean(dire_performance), num_privates


def add_to_db(match):
    match_id = match['match_id']
    urls = get_players_urls(get_acc_list(match_id), match_id)
    rad_performance, dire_performance, num_privates = get_players_performance(urls, match_id)
    db_collection.update_one({'match_id': match_id},
                             {'$set': {'rad_performance': rad_performance, 'dire_performance': dire_performance,
                                       'num_privates': num_privates}},
                             upsert=True)


if __name__ == '__main__':
    matches = db_collection.find({"rad_performance" : None}).sort([["match_id", -1]])

    # Define number of cores to run the loop parallelly
    num_cores = 3
    with multiprocessing.Pool(num_cores) as p:
        p.map(add_to_db, matches)
    p.close()
    p.join()
