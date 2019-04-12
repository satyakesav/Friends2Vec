import os
import random

cwd = os.getcwd()
ratingsFile = cwd + '/ratings.txt'
userDict = {}
with open(ratingsFile, 'r') as infile:
    for line in infile:
        #print(line)
        if line.strip():
            userId, restroId, rating = line.split('\t')
            #print(userId, restroId, rating)
            if userDict.get(userId) == None:
                userDict[userId] = [(restroId, rating)]
            else:
                userDict[userId].append((restroId,rating))


print(len(userDict))
trainFile = cwd +'/yelp.train.rating'
testFile = cwd +'/yelp.test.rating'
with open(trainFile, 'w') as train, open(testFile, 'w') as test:
    for user in userDict.keys():
        userList = userDict[user]
        random.shuffle(userList)
        for i in range(len(userList)):
            data = user + '\t' + userList[i][0] + '\t' + userList[i][1]
            #print(data)
            if i == 0:
                test.write(data)
            else:
                train.write(data)
