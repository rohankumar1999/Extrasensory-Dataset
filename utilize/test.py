from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, M_test=None, report=False):
    
    y_pred = model.predict(X_test)
    
    if M_test is not None:
        sample_weight = M_test
    else:
        sample_weight = None
    
    accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int), sample_weight=sample_weight)
    precision = precision_score(y_test, (y_pred > 0.5).astype(int), average='macro', sample_weight=sample_weight)
    recall = recall_score(y_test, (y_pred > 0.5).astype(int), average='macro', sample_weight=sample_weight)
    f1 = f1_score(y_test, (y_pred > 0.5).astype(int), average='macro', sample_weight=sample_weight)
    
    if report:
        print(f"{'Metric':<15}{'Score':<15}")
        print(f"{'Accuracy':<15}{accuracy:<15.6f}")
        print(f"{'Precision':<15}{precision:<15.6f}")
        print(f"{'Recall':<15}{recall:<15.6f}")
        print(f"{'F1 Score':<15}{f1:<15.6f}")
    
    return accuracy, precision, recall, f1



