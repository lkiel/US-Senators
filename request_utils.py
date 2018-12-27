import requests
import json

API_KEY = ""

with open('api_key.txt') as f:
    API_KEY = f.read()

HEADERS = {"X-Api-Key": API_KEY}


def print_info(id):

    bill_id, congress, roll_call, session = id.split('-')

    bill_url = "https://api.propublica.org/congress/v1/115/bills/{bill_id}.json"
    vote_url = "https://www.senate.gov/legislative/LIS/roll_call_lists/roll_call_vote_cfm.cfm?congress=115&session={}&vote={:05d}"
    res = requests.get(url = bill_url.format(bill_id=bill_id), headers = HEADERS)
    
    jObj = json.loads(res.text)['results'][0]
    
    print("\033[1m============== INFORMATION REGARDING " + str(id) + " ============== \033[0m")
    print("\n\033[1mRoll call:\033[0m " + roll_call)
    print("\033[1mSession:\033[0m " + session)
    print("\n"+vote_url.format(session, int(roll_call)))
    print("\n\033[1mRELATED BILL INFORMATION\033[0m\n")
    print("\033[1mBill title: \033[0m" + str(jObj['title']))
    print("\n\033[1mBill url: \033[0m " + str(jObj['govtrack_url']))
    print("\n\033[1mShort description: \033[0m" + str(jObj['summary_short'])+"\n\n\n")