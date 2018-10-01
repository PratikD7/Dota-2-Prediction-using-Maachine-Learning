import threading
import dota2api
import pickle

D2_API_KEY = 'A628D1DB395B88A8D0E1999FBCF8DB59';  # CAREFUL while uploading to github #
api = dota2api.Initialise(D2_API_KEY)
list_of_ids = [4134697476, 4124697476, 4114697476, 4104697476, 4094697476, 4084697476, 4074697476, 4064697476, 4054697476, 4044697476, 4034697476, 4024697476, 4014697476,
               4004697476]


class myThread(threading.Thread):
    def __init__(self, start_id, end_id, thread_id):
        threading.Thread.__init__(self)
        self.start_id = start_id
        self.end_id = end_id
        self.thread_id = thread_id

    def run(self):
        getMatchDetailsByID(self.start_id, self.end_id, self.thread_id)


def getMatchDetailsByID(match_id_start, match_id_end, thread_id):
    for i in range(match_id_end, match_id_start):
        print("Thread id", thread_id)
        print("Remaining matches data to pull", match_id_start - i)
        print()
        try:
            match = api.get_match_details(match_id=i)
            if (match['duration'] >= 900 and match['lobby_type'] == 7 and (match['game_mode'] == 1 or match['game_mode'] == 22)):
                f = open('D:/MLG/Dota2-SL/Match Data Files/' + str(i) + '.txt', 'wb')
                pickle.dump(match, f)
                # f.write(str(match))
                f.close()
        except:
            pass

# Create new threads
thread1 = myThread(list_of_ids[0], list_of_ids[1], 1)
thread2 = myThread(list_of_ids[1], list_of_ids[2], 2)
thread3 = myThread(list_of_ids[2], list_of_ids[3], 3)
thread4 = myThread(list_of_ids[3], list_of_ids[4], 4)
thread5 = myThread(list_of_ids[4], list_of_ids[5], 5)
thread6 = myThread(list_of_ids[5], list_of_ids[6], 6)
thread7 = myThread(list_of_ids[6], list_of_ids[7], 7)
thread8 = myThread(list_of_ids[7], list_of_ids[8], 8)
thread9 = myThread(list_of_ids[8], list_of_ids[9], 9)
thread10 = myThread(list_of_ids[9], list_of_ids[10], 10)


# Start new Threads
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
thread8.start()
thread9.start()
thread10.start()
print("Exiting Main Thread")
