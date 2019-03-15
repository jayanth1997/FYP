from nltk.stem import LancasterStemmer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#lemmatiser = WordNetLemmatizer()
file1 = open("train.txt")
#file2 = open("test.json")
for line in iter(file1):

	#lines = file1.readline()
	#while lines:
	#line = file1.read()# Use this to read file content as a stream:
	  words = line.split()
	  for r in words:
			   m=stemmer.stem(r)
			#if not r in stop_words and r!=" ":
			   appendFile = open('stemtrain.json','a')
			   #appendFile = open('testing.json','a')
			   appendFile.write(" "+m)
	  appendFile.write('\n')		   
	  appendFile.close()	
#print("Stem %s: %s" % ("studying", stemmer.stem("studying")))
#print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studying")))
#print("Lemmatise %s: %s" % ("studying", lemmatiser.lemmatize("studying", pos="v")))
