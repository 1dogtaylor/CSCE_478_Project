from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.cluster import KMeans

path = "/Users/taylorbrandl/Desktop/Taylor/Python/ML_Project/Labeled_images/"

mode = 'run'

data_size = 0
#preprocess the data
class preprocess:
    def __init__(self):
        pass

    def read_data(self,path):
        imgMat = np.array(np.zeros(217088))
        
        for filename in os.listdir(path):
    
            if filename.endswith(".png"): # to only get depth images
                #image path
                imgpath = path + filename
                # Read the image
                image = Image.open(imgpath)
                dataArray = np.array(image)
                #normalize
                Norm_dataArray = (dataArray - np.amin(dataArray))/(np.amax(dataArray) - np.amin(dataArray))
                #masking
                masked_img = masking.mask(Norm_dataArray)
                #masked_img = Norm_dataArray

                #plt.imshow(masked_img)     #Show Image
                #plt.show()
                
                #flatten the array
                data = masked_img.flatten()
                #use np.vstack() to stack the vectors vertically
                imgMat = np.vstack([imgMat,data])
        data_size = len(imgMat)
        return imgMat, data_size

    def split_data(self, imgMat, gt, n=0.8):
        #split the data into training and testing
        #training data
        index = int(n*len(imgMat))
        trainData = imgMat[0:index, :]
        trainLabels = gt[0:index]
        #testing data
        testData = imgMat[index+1:, :]
        testLabels = gt[index+1:]
        return trainData, testData, trainLabels, testLabels

    def read_labels(self,path,data_size):
        #read in the labels
        gt = []
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                file = pd.read_csv(path + filename, sep=",", header=0)
                ground_truth = file['postion']
        for label in ground_truth:
            if label == 'Standing':
                gt.append(0)
            elif label == 'Sitting':
                gt.append(1)
            elif label == 'Kneeling':
                gt.append(2)
            elif label == 'Lying sternum':
                gt.append(3)
            elif label == 'Udder away from heat lamp' or label == 'Udder towards heat lamp' or label == 'Lying other':
                gt.append(4)
        return gt[:data_size]

class masking:
    def mask(dataArray):
        mask = np.array(np.zeros(217088))
        mask = mask.reshape(424,512)
        for x in range(0, 512):
            for y in range(0, 424):
                if y > -0.45*x+273 and y < -0.45*x+416 and y > 2.222*x-900 and y < 2.2222*x+30:
                    mask[y,x] = 1
                else:
                    mask[y,x] = 0
        masked_img = dataArray*mask
        cropped_img = masked_img#[masked_img != 0]
        return cropped_img
    

class Kmeans:
    def __init__(self, train_data):
        self.imgMat = train_data
    def cluster(self, n_clusters):
        #kmeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(self.imgMat)
        #print("cluster centers",kmeans.cluster_centers_)
        #print("kmean labels",kmeans.labels_)
        return kmeans

class test:
    def __inti__(self):
        pass
        

    def return_predictions(self, kmeans, test_data, test_labels):
        correct = []
        predictions = kmeans.predict(test_data)
        for i in range(len(test_data)):
            if predictions[i] == test_labels[i]:
                correct.append(1)
            else:
                correct.append(0)
        
        print("correct",correct)
        accuracy = sum(correct)/len(correct)
        print("accuracy", accuracy)
        return accuracy
        
data, data_size = preprocess().read_data(path)
gt = preprocess().read_labels(path,data_size)
print("data shape",data.shape)
train_data, test_data, train_labels, test_labels = preprocess().split_data(data,gt)

if mode == 'train':        
    kmeans = Kmeans(train_data).cluster(5)
    dump(kmeans, 'kmeans_5.joblib')
    accuracy = test().return_predictions(kmeans, test_data, test_labels)
    print("accuracy",accuracy)
elif mode == 'run':
    kmeans = load('kmeans_5.joblib')
    accuracy = test().return_predictions(kmeans, test_data, test_labels)
    print("accuracy",accuracy)






