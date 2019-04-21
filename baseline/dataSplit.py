import os
import random

cwd = os.getcwd()
cwd = "./data/yelp2"
ratingsFile = cwd + '/ratings.txt'
#print(ratingsFile)
userDict = {}
ctr = 0
with open(ratingsFile, 'r') as infile:
	print("enter")
	for line in infile:
		ctr += 1
		#print(line)
		if line.strip():
			userId, restroId, rating = line.split('\t')
			if userDict.get(userId) == None:
				userDict[userId] = [(restroId, rating)]
			else:
				userDict[userId].append((restroId,rating))


print(len(userDict))
print(ctr)
trainFile = cwd +'/yelp.train.rating'
testFile = cwd +'/yelp.test.rating'
validFile = cwd + '/yelp.valid.rating'
with open(trainFile, 'w') as train, open(testFile, 'w') as test, open(validFile, 'w') as valid:
	for user in userDict.keys():
		userList = userDict[user]
		random.shuffle(userList)
		for i in range(0, int(0.6 * len(userList))):
			data = user + '\t' + userList[i][0] + '\t' + userList[i][1]
			train.write(data)
		for i in range( int(0.6 * len(userList)), int(0.8 * len(userList))):
			data = user + '\t' + userList[i][0] + '\t' + userList[i][1]
			valid.write(data)
		for i in range( int(0.8 * len(userList)), len(userList)):
			data = user + '\t' + userList[i][0] + '\t' + userList[i][1]
			test.write(data)
