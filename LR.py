
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler   
from sklearn.linear_model import LogisticRegression  
from matplotlib.colors import ListedColormap  
from sklearn.metrics import confusion_matrix  
#importing datasets  

data_set= pd.read_csv('Social_Network_Ads.csv')  
#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [0,1]].values  
y= data_set.iloc[:, 2].values 


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=4)  

# Feature Scaling   
# StandardScaler chia tỉ lệ tương đối dữ liệu 
# giải thích tiếng việt cho dễ hiểu : 
# Phương thức Fit (): tính toán các tham số và và lưu chúng dưới dạng đối tượng bên trong.
# Transform (): Phương thức sử dụng các tham số được tính toán này áp dụng phép chuyển đổi cho một tập dữ liệu cụ thể.
# Fit_transform (): tham gia phương thức fit () và Transform () để chuyển đổi tập dữ liệu.
st_x= StandardScaler()   
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

#Fitting Logistic Regression to the training set  
classifier= LogisticRegression(random_state=4)  
classifier.fit(x_train, y_train) 

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,  
                   intercept_scaling=1, l1_ratio=None, max_iter=100,  
                   multi_class='warn', n_jobs=None, penalty='l2',  
                   random_state=4, solver='warn', tol=0.0001, verbose=0,  
                   warm_start=False)  


#Visualizing the training set result  
# sample 
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  

#Visulaizing the test set result  
# realize 
x_set, y_set = x_test, y_test  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Testing set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  



# 