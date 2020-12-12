import facial_features
import time
import os
from PIL import Image
import numpy as np 
import scipy
import LSB
import shutil
import config

def main():
    choice = 0
    while choice != '1' and choice != '2' and choice != '3' and choice != '4': 
        choice = input("Choose 1 to encode or 2 to decode or 3 to encode many or 4 to decode many: ")

    if choice == '1':
        menuEncode()
    elif choice == '3':
        multiEncode()
    elif choice == '4':
        multiDecode()
    else:
        menuDecode()


def multiEncode():
    for i in range(28):
        picture, imgPath = facial_features.select_image(f'{i}.png')
        chosenFeature, pointsList, pixelsList = facial_features.do_facial_feature_recog(picture, imgPath)
        print("The important information: \n Picture chosen: {} \n Chosen feature: {} ".format(picture, chosenFeature))
        LSB.encode(picture,imgPath,pointsList,pixelsList,f'{i}.png')

def multiDecode():
    for i in range(28):
        #picture = str(input("Enter the image with extension (ex: example.png): "))
        print("This is the current path: ", os.getcwd())
        #imgPath = str(input("Enter the path to the image: "))
        #facialFeature = str(input("Enter the facial feature (eyes, mouth, nose): "))
        #we are passing 1 in to the facial_feature_recog function to tell that function to decode
        picture = f'{i}.png'
        imgPath = os.path.join(os.getcwd(), "facial_recog", "dataset", picture)
        toGetPoints = os.path.join(os.getcwd(), "facial_recog", "original_dataset", picture)
        facialFeature = config.facial_feature
        pointsList = facial_features.do_facial_feature_recog(picture, imgPath, 1, facialFeature)
        pointsList = pointsList[1]
        try:
            print("Decoded: {}".format(LSB.decode(picture,imgPath, pointsList)))
            print("Picture: {}".format(picture))
            #shutil will copy the original file and replace the encoded image with the original copy of the image
            # shutil.copyfile(toGetPoints, imgPath)  
        except StopIteration:
            print("Exception")
            shutil.copyfile(toGetPoints, imgPath)


def menuEncode():
    picture, imgPath = facial_features.select_image(config.picture)
    chosenFeature, pointsList, pixelsList = facial_features.do_facial_feature_recog(picture, imgPath)
    print("The important information: \n Picture chosen: {} \n Chosen feature: {} ".format(picture, chosenFeature))
    LSB.encode(picture,imgPath,pointsList,pixelsList,config.picture)

def multiEncode():
    i = 1
    path = os.path.join(os.getcwd(), "facial_recog", "original_dataset")
    while i < 21:
        picture = str(i)+".png" #test
        os.chdir(path)
        imgPath = path+"/"+picture
        chosenFeature, pointsList, pixelsList = facial_features.do_facial_feature_recog(picture, imgPath)
        print("The important information: \n Picture chosen: {} \n Chosen feature: {} ".format(picture, chosenFeature))
        LSB.encode(picture,imgPath,pointsList,pixelsList,str(i))
        i = i + 1

def multiDecode():
    i = 1
    print("This is the current path: ", os.getcwd())
    while i < 21:
        picture = str(i)+'.png'
        imgPath = os.path.join(os.getcwd(), "facial_recog", "dataset", picture)
        toGetPoints = os.path.join(os.getcwd(), "facial_recog", "original_dataset", picture)
        facialFeature = 'nose'
        pointsList = facial_features.do_facial_feature_recog(picture, imgPath, 1, facialFeature)
        pointsList = pointsList[1]
        try:
            print("Decoded: {}".format(LSB.decode(picture,imgPath, pointsList)))
            #shutil will copy the original file and replace the encoded image with the original copy of the image
            # shutil.copyfile(toGetPoints, imgPath)  
        except StopIteration:
            print("Exception")
            shutil.copyfile(toGetPoints, imgPath)
        i = i + 1

def menuDecode():
    #picture = str(input("Enter the image with extension (ex: example.png): "))
    print("This is the current path: ", os.getcwd())
    #imgPath = str(input("Enter the path to the image: "))
    #facialFeature = str(input("Enter the facial feature (eyes, mouth, nose): "))
    #we are passing 1 in to the facial_feature_recog function to tell that function to decode
    picture = config.picture
    imgPath = os.path.join(os.getcwd(), "facial_recog", "dataset", picture)
    toGetPoints = os.path.join(os.getcwd(), "facial_recog", "original_dataset", picture)
    facialFeature = config.facial_feature
    pointsList = facial_features.do_facial_feature_recog(picture, imgPath, 1, facialFeature)
    pointsList = pointsList[1]
    try:
        print("Decoded: {}".format(LSB.decode(picture,imgPath, pointsList)))
        #shutil will copy the original file and replace the encoded image with the original copy of the image
        # shutil.copyfile(toGetPoints, imgPath)  
    except StopIteration:
        print("Exception")
        shutil.copyfile(toGetPoints, imgPath)

main()