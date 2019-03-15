import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#word_tokenize accepts a string as an input, not a file.
stop_words = set(stopwords.words('english'))
file1 = open("test2pre.txt")
#file2 = open("art.txt")
for line in iter(file1):

	#lines = file1.readline()
	#while lines:
	#line = file1.read()# Use this to read file content as a stream:
	  words = line.split()
	  for r in words:
			if not r in stop_words and r!=" ":
			   appendFile = open('testset2.txt','a')
			   #appendFile = open('testing.json','a')
			   appendFile.write(" "+r)
	  appendFile.write('\n')		   
	  appendFile.close()	