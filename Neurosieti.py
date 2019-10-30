import pandas as pd
import csv
import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

Data = pd.read_csv('C:/test/input.csv')
Data.head (10)
Data['black_rating']
print (Data[['turns', 'white_rating']][:10])
print(Data['black_rating'])
train_size = int(0.75 * Data.shape[0])
test_size = int(0.25 * Data.shape[0])
print ("Размер обучающего набора:" + str (train_size))
print ("Тестирование установленный размер: "+ str (test_size))

Data = Data.sample(frac= 1 )
X = Data.iloc [:, [10, 10 ]]
y = Data.iloc [:, 4]
X = X.atype (float)
X_train=X[0:train_size,:]
y_train=y[0:train_size]

X_test=X[train_size:,:]
y_test=y[train_size:]

X_set, y_set = X_train, y_train
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('white', 'black'))(i), label = j,marker='.')

plt.title('Training set')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

def generate_data( class_data_dic ,  X_train, y_train):
    first_one = True
    first_zero = True

    for i in range(y_train.shape[0]):
                X_temp=X_train[i,:].reshape(X_train[i,:].shape[0],1)
                if y_train[i]==1:
                    if first_one==True:
                        class_data_dic[1]=X_temp
                        first_one=False
                    else:
                        class_data_dic[1]= np.append(class_data_dic[1],X_temp,axis=1)

                elif y_train[i]==0:
                    if first_zero==True:
                        class_data_dic[0]=X_temp
                        first_zero=False
                    else:
                        class_data_dic[0]= np.append(class_data_dic[0],X_temp,axis=1)
                np.append(class_data_dic[0], X_temp, axis=1)
    return class_data_dic
    mean_0 = np.mean(class_data_dic[0], axis=0)
    mean_1 = np.mean(class_data_dic[1], axis=0)
    std_0 = np.std(class_data_dic[0], axis=0)
    std_1 = np.std(class_data_dic[1], axis=0)


    def likelyhood(x, mean, sigma):
                return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2)) * (1 / (np.sqrt(2 * np.pi) * sigma))

    def posterior(X, X_train_class, mean_, std_):
        product = np.prod(likelyhood(X, mean_, std_), axis=1)
        product = product * (X_train_class.shape[0] / X_train.shape[0])
        return product

        p_1 = posterior(X_test, class_data_dic[1], mean_1, std_1)
        p_0 = posterior(X_test, class_data_dic[0], mean_0, std_0)
        y_pred = 1 * (p_1 > p_0)
        print(y_pred)