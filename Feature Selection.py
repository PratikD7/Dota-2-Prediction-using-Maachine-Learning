"""
Data Cleaning
Features : radiant_win,duration,[kda_ratio,gpm+xpm,level,hero+tower_damage,
gold_spent,heroes winrate ,pickrate, kda]of all 10 players = 32 features
"""
import pickle
import os
import pandas as pd

total_matches = 103197

hero_data = pd.read_csv('Hero-win-rates.csv').values

path = 'D:/MLG/Dota2-SL/Match Data Files/'
hero_ids = []
f3 = open('intr_features.csv', 'a')
count = 0
# Scrap the data from the RAW data files
for f in os.listdir(path):
    count += 1
    print("Percentage complete:-", count / total_matches * 100, "%")

    hlist = []
    f2 = open(path + f, 'rb')
    data = pickle.load(f2)
    strg = ""
    if 'radiant_win' in data:
        strg = strg + str(data['radiant_win']) + ','
    else:
        strg += '?,'
    if 'duration' in data:
        strg = strg + str(data['duration']) + ','
    else:
        strg += '?,'
    if 'players' in data:
        for player in data['players']:

            if 'hero_id' in player:
                hlist.append(player['hero_id'])
            else:
                strg = '0,'

            if 'kills' in player and 'deaths' in player and 'assists' in player:
                # KDA ratio
                try:
                    kda = (player['kills'] + player['assists']) / player[
                        'deaths']
                except ZeroDivisionError:
                    kda = kda = (player['kills'] + player['assists'])
                strg = strg + str(kda) + ','
            else:
                strg += '?,'

            if 'gold_per_min' in player and 'xp_per_min' in player:
                strg = strg + str(
                    player['gold_per_min'] + player['xp_per_min']) + ','
            else:
                strg += '?,'
            print(strg)
            if 'level' in player:
                strg = strg + str(player['level']) + ','
            else:
                strg += '?,'
            if 'hero_damage' in player and 'tower_damage' in player:
                strg = strg + str(
                    player['hero_damage'] + player['tower_damage']) + ','
            else:
                strg += '?,'
            if 'gold_spent' in player:
                strg = strg + str(player['gold_spent']) + ','
            else:
                strg += '?,'
    else:
        strg += '?,'

    hero_ids.append(hlist)
    for h in hlist:
        for row in hero_data:
            if h == row[-1]:
                strg += str(row[1]) + ',' + str(row[3]) + ','
    strg = strg[:-1]
    f3.write(strg + '\n')
    f2.close()
f3.close()

'''
from sklearn.preprocessing import MultiLabelBinarizer

# Create MultiLabelBinarizer object
one_hot = MultiLabelBinarizer()

# One-hot encode data
ohearray = (one_hot.fit_transform(hero_ids))
ohearray = ohearray.tolist()
# print(len(ohearray))
# print(one_hot.classes_)

# f4 = open('intr_features.csv', 'r')
# f5 = open('final_features.csv', 'a')
# i = 0
# for line in f4:
#     # print(ohearray[i])
#     lines = line[:-1] + str(ohearray[i]) + '\n'
#     f5.write(lines)
#     i += 1
# f4.close()
# f5.close()
'''


f4 = open('intr_features.csv', 'r')
f5 = open('final_features.csv', 'a')
i = 0
for line in f4:
    lines = line[:-1] + '\n'
    f5.write(lines)
    i += 1
f4.close()
f5.close()
