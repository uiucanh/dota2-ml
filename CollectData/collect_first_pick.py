import requests

from src.CollectData.collect_data import setup_db_log

# Collect information on which team got first pick and last pick
def get_picks_order(match):
    first_pick = 0
    last_pick = 0

    for pick in match['pickBans']:
        if pick['isPick']:              # Ignore bans
            if pick['order'] == 0:
                first_pick = 1 if pick['isRadiant'] else -1
            if pick['order'] == 9:
                last_pick = 1 if pick['isRadiant'] else -1
    return first_pick, last_pick

if __name__ == "__main__":
    db_collection, logger = setup_db_log('collect_first_pick.log')

    while True:
        matches = db_collection.find({"First_Pick" : None}).limit(50)
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
                    first_pick, last_pick = get_picks_order(match)
                except KeyError:
                    print(1)
                    db_collection.update_one({'match_id': match_id}, {"$set": {'First_Pick' : 0}}, upsert=True)
                else:
                    db_collection.update_one({'match_id': match_id}, {"$set": {'First_Pick' : first_pick}}, upsert=True)
                    db_collection.update_one({'match_id': match_id}, {"$set": {'Last_Pick' : last_pick}}, upsert=True)

                logger.info("Finished processing match: %s" % match_id)