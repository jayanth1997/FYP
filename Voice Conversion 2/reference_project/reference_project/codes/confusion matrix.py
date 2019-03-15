from pandas_ml import ConfusionMatrix
#binary_confusion_matrix.plot(backend=Backend.Seaborn)
#binary_confusion_matrix.plot(backend='seaborn')
#binary_confusion_matrix.plot(backend=Backend.Seaborn)
file1 = open("test2actualf.txt")
file2 = open("boost.txt")
y_true = []
y_pred = []
for line in iter(file1):
	    y_true.append(line)
		
for line in iter(file2):
	    y_pred.append(line)
#y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
#y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
cm = ConfusionMatrix(y_true, y_pred)
#binary_confusion_matrix = BinaryConfusionMatrix(y_true, y_pred)
print("Confusion matrix:\n%s" % cm.stats)
#cm = ConfusionMatrix(y_true, y_pred)
#appendFile = open('naivede.txt','a')
#appendFile.write(cm.stats())
cm.print_stats()
#accuracy_score(y_true, y_pred)