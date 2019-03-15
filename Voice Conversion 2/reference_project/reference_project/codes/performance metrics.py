from mlxtend.evaluate import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score


#y_target =    [1, 1, 1, 0, 0, 2, 0, 3]
#y_predicted = [1, 0, 1, 0, 0, 2, 1, 3]
file1 = open("test2actualf.txt")
file2 = open("boost.txt")
y_target = []
y_predicted = []
for line in iter(file1):
	    y_target.append(line)
		
for line in iter(file2):
	   y_predicted.append(line)
		
#cm = confusion_matrix(y_true, y_pred)	
#print("Confusion matrix:\n%s" % cm)	
#df_confusion = pd.crosstab(y_actu, y_pred)
cm = confusion_matrix(y_target=y_target, 
                      y_predicted=y_predicted, 
                      binary=False)
print("Confusion matrix:\n%s" % cm)	
precision, recall, fscore, support = score(y_target,y_predicted)
print('precision: \n {}'.format(precision))
print('recall:{}'.format(recall))
print('fscore:{}'.format(fscore))
#m=accuracy_score(y_target, y_predicted)
                     
#print("accuracy:\n%s" % m)