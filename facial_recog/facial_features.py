from PIL import Image, ImageDraw
import os
import face_recognition
import time
import random
import config

currentWorkingDirectory = os.getcwd()

def select_image(imgNum):
    #replace this with a prompt for user to input the path to dataset
    path = os.path.join(currentWorkingDirectory, "facial_recog", "original_dataset")
    pictures_list = os.listdir(path)
    #randomly selecting a element from our picture list to perform facial feature recognition 
    #picture = str(input("Enter the name of the file you want to use for encoding: "))
    picture = f'{imgNum}' #test
    os.chdir(path)
    img_path = path+"/"+picture
    return picture, img_path

def formatEncodingBox(startingPoint, imageSize):
    middleOfPicture = imageSize[0]/2
    boxDecider = startingPoint[0] - middleOfPicture
    if boxDecider >= 0:
        return [startingPoint[0] - 150, startingPoint[1]]
    else:
        return [startingPoint[0] + 150, startingPoint[1]]

def do_facial_feature_recog(img,path, decode = 0, facialFeature = None):
        image = face_recognition.load_image_file(path)
        face_landmarks_list = face_recognition.face_landmarks(image)
        pil_image = Image.fromarray(image)
        d = ImageDraw.Draw(pil_image)

        for face_landmarks in face_landmarks_list:
            #combining bottom lip, top lip, and chin into mouth
            #combining left_eye, left_eyebrow, right_eye, right_eyebrow into eyes
            #combing nose_bridge, nose_tip into nose.
            face_landmarks['mouth'] = face_landmarks['bottom_lip'] + face_landmarks['top_lip'] + face_landmarks['chin']
            face_landmarks['eyes'] = face_landmarks['left_eye'] + face_landmarks['right_eye'] + face_landmarks['left_eyebrow'] + face_landmarks['right_eyebrow']
            face_landmarks['nose'] = face_landmarks['nose_tip'] + face_landmarks['nose_bridge']
            #cleaning up the leftover points
            toRemove = ["bottom_lip","top_lip","chin","left_eye","right_eye","left_eyebrow","right_eyebrow","nose_bridge","nose_tip"]
            for each in toRemove:
                face_landmarks.pop(each)

            if decode == 1:
                facial_feature = facialFeature
            else: 
                #facial_feature = random.choice(list(face_landmarks.keys()))
                #facial_feature = str(input("Enter the facial feature that you want to use for encoding: "))
                facial_feature = config.facial_feature # delete this later

            baselines = face_landmarks[facial_feature]
            points = []
            #this is to increase our points selections
            lengthOfPoints = len(baselines[0]) - 1
            print("This is length of points before expanding: ", lengthOfPoints) #test
            
            x, y = baselines[0]
            x, y = formatEncodingBox([x,y], pil_image.size)
            if x > pil_image.size[0] / 2:
                for j in range(-10,10):
                    for k in range(-10,10):
                        points.append((x+j, y+k))
            else:
                for j in range(-10,10):
                    for k in range(-10,10):
                            points.append((x-j, y-k))

            #removing duplicates
            points = list(dict.fromkeys(points))
            print("This is len of points: ", len(points)) #test
            #Extracting pixel values
            pixels = pil_image.load()
            pixel_list = []
            for pair in points:
                x,y = pair[0], pair[1]
                pixel_list.append(pixels[x,y])
        d.point(points) #test
        # pil_image.show() #test
        return facial_feature,points,pixel_list

