from __future__ import print_function
import os, sys, pickle
import pandas as pd
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import cross_val_score
from sklearn import svm


def train(flared_folder, good_folder, filename):
    
    """ Trains a SVC using images from the flared_folder and good_folder
    saving the resulting model in the filename
    
    Args:
        flared_folder (str): folder to find the flared images
        good_folder (str): folder to find the good images

    """
    
    print("Getting photos.")
    
    #get the files from the directories
    #and give them a label based on directory
    train_f, y_f = get_files(flared_folder, 1)
    train_g, y_g = get_files(good_folder, 0)
    
    #append the flared and good lists together
    train_files = train_f + train_g
    y = y_f + y_g
    
    print("Found %d flared photos and %d good photos." % (len(train_f), len(train_g)))
    print("Building features.")
    
    #build the features
    X = build_hog_features(train_files)
    
    print("Training model.")
    #build the Support Vector Classifier
    model = svm.SVC(gamma="scale")
    
    print("Performing cross validation.")
    
    #run a cross validation
    a = cross_val_score(model, X, y, cv=5)
    
    print("Model trained with cross val score of %.2f"% a.mean())
    
    print("Fitting model.")
    model.fit(X, y)

    print("Saving model.")
    try:
        pickle.dump(model, open(filename, 'wb'))
    except OSError:
        print("Could not save model.")
        return
    
    print("Model saved to %s"% filename)

def predict(filenames):
    
    """ Prints a prediction to stdout for each of the images
    prints
    1 for a flared image
    0 for a good image
        
    Args:
        filenames (list[str]): list of files to predict
        
    """
    
    try:
        model = pickle.load(open("default.sav", 'rb'))
    except OSError:
        print("Could not load model.")
        return
    
    X = build_hog_features(filenames)
    
    pred = model.predict(X)
    
    for p in pred:
        print(p)
        


def get_files(folder, label):
    """ Extracts the filenames from the passed folders and gives them all the
    same label for the ML algo
    
    Args:
        folder (str): folder to find files
        label (int): ML label to give all files from this folder
        
    Returns:
        list(str): list of the relative filepaths/filenames
        list[int]: corresponding list of the labels for the images
    """
    
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    train_files = []
    y_train = []
    for _file in files:
        train_files.append(os.path.join(folder,_file))
        y_train.append(label)     
        
    return train_files, y_train

def build_hog_features(train_files):
    """ A function to build ML features from the skimage hog function
    builds totol zero hog gradient, max hog gradient and mean hog gradient
    from a list of training files
    
    Args:
        train_files (list[str]): list of files to analyse
        
    Returns:
        DataFrame: a dataframe containg the features
    """
    
    #lists to store hog features for each image
    maxes = []
    means = []
    count_zero = []

    #loop through the train files , load the image
    #run the hog function and get
    #max hog values, mean hog values and count the zero gradient
    #hog values
    for i in range(len(train_files)):
        #load image and resize it for th hog function
        flared = imread(train_files[i])
        resized = resize(flared, (64,128))

        #run the hog feature
        fd, hog_img = hog(resized, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True, multichannel=True)
        
        #calculate max, mean and count zero gradients
        maxes.append(fd.max())
        means.append(fd.mean())
        count_zero.append(len(fd[fd == 0]))
    
    #create a dataframe of the features to return
    features = pd.DataFrame()
    features["hog_mean"] = means
    features["hog_max"] = maxes
    features["hog_zero_count"] = count_zero
    
    return features
    

if __name__ == "__main__":
    
    #example train commandline
    #train training/flare training/good default.sav

    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            if len(sys.argv) != 5:
                print("You must supply flared directory and good directory as args 2 and 3 and save model filename as arg 4/", file=sys.stderr)
            else:
                flared_folder, good_folder, filename = sys.argv[2], sys.argv[3], sys.argv[4]
                train(flared_folder, good_folder, filename)
        else:
            predict(sys.argv[1:])
            
