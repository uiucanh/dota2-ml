import requests
import numpy as np

from src.CollectData.collect_data import setup_db_log

# Collect the pick order of each hero in the match
def get_picks_order(match):
    radiant_picks = []
    dire_picks = []

    for pick in match['pickBans']:
        if pick['isPick']:              # Ignore bans
            if pick['isRadiant']:
                radiant_picks.append(pick['heroId'])
            else:
                dire_picks.append(pick['heroId'])

    return radiant_picks, dire_picks

if __name__ == "__main__":
    db_collection, logger = setup_db_log('collect_picks_order.log')

    while True:
        matches = db_collection.find({"Rad_Pick_0" : None}).limit(150).sort([["match_id", -1]])
        if matches.count() == 0:
            logger.info("No more matches left")
            exit()
        else:
            # A list of all the query matches
            match_list = []
            for match in matches:
                match_list.append(match['match_id'])

            # Create url to call Stratz API
            match_url = ""
            for match in range(len(match_list)):
                if match == 0:
                    match_url += str(match_list[match])
                else:
                    match_url += ',' + str(match_list[match])

            url = "https://api.stratz.com/api/v1/match?matchID=" + match_url + "&include=PickBan&take=250"
            queries = requests.get(url).json()

            for match in queries['results']:
                match_id = match['id']

                try:
                    radiant_picks, dire_picks = get_picks_order(match)
                except KeyError:
                    logger.error("Match %s could not pre processed" % match_id)
                    db_collection.update_one({'match_id': match_id}, {"$set": {"Rad_Pick_0": np.nan}},
                                             upsert=True)
                    continue
                else:
                    for i in range(0,5):
                        rad_hero = "Rad_Pick_" + str(i)
                        db_collection.update_one({'match_id': match_id}, {"$set": {rad_hero : radiant_picks[i]}}, upsert=True)

                    for i in range(0,5):
                        dire_hero = "Dire_Pick_" + str(i)
                        db_collection.update_one({'match_id': match_id}, {"$set": {dire_hero : dire_picks[i]}}, upsert=True)

                    logger.info("Finished processing match: %s" % match_id)