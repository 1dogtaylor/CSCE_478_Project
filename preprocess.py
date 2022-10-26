from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

path = "/Users/taylorbrandl/Desktop/Taylor/Python/ML_Project/Test_Images/"

#preprocess the data
class preprocess:
    def __init__(self, path):
        self.path = path


    def read_data(self):
        imgMat = np.array(np.zeros(217088))
        
        for filename in os.listdir(self.path):
    
            if filename.endswith(".png"): # to only get depth images
                #image path
                imgpath = path + filename
                # Read the image
                image = Image.open(imgpath)
                dataArray = np.array(image)
                #normalize
                dataArray = dataArray/255
                #masking
                #masked_img = masking.mask(dataArray)
                masked_img = dataArray
                plt.imshow(masked_img)
                plt.show()
                
                #flatten the array
                data = masked_img.flatten()
                #use np.vstack() to stack the vectors vertically
                imgMat = np.vstack([imgMat,data])
                
            
        
        return imgMat

    

    def split_data(self, imgMat, n=0.8):
        #split the data into training and testing
        #training data
        index = int(n*len(imgMat))
        trainData = imgMat[1:index, :]
        #testing data
        testData = imgMat[index+1:, :]
        return trainData, testData

class masking:
    def mask(dataArray):
        mask = np.array(np.zeros(217088))
        mask = mask.reshape(512,424)
        for x in range(0, 512):
            for y in range(0, 424):
                if y < 0.5*x+131 and y > 0.5*x+8.65 and y < -1.77*x+1084.4 and y > -2.5*x+431:
                    mask[x,y] = 1
                else:
                    mask[x,y] = 0
        
        return dataArray*mask
    

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
        

    def return_predictions(self, kmeans, test_data):
        return kmeans.predict(test_data)
        
data = preprocess(path).read_data()
print("data shape",data.shape)
train_data, test_data = preprocess(path).split_data(data)
kmeans = Kmeans(train_data).cluster(5)
predictions = test().return_predictions(kmeans, test_data)
print("preditions",predictions)






