import classifier
import translator
import concurrent.futures
import multiprocessing
import log
import progressbar

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("-------------------------------------------------------------")

#show options to user
print("Classification 1")
print("Translation 2")
taskPick = int(input("Please pick a task: "))


#image classfication
if taskPick == 1:
    
    grey = input("Greyscale Images y/n: ")
    
    classif = classifier.classifier(grey)
    print("1. KNN")
    print("2. GNB")
    print("3. CNN")
    classTag = int(input("Please Enter Classifier to use:"))
    classif.BuildModel(classTag)


    #test data
    log.start("Testing...")
    #spawn threads to go through and test the data
    threadCount = multiprocessing.cpu_count()
    #classif.testData(0, "cat")
    with concurrent.futures.ThreadPoolExecutor(max_workers=threadCount) as executor:
        for index in range(500):
            executor.submit(classif.testData, index, "dog")
    log.stop("Testing Finished")

    print("-------------------------------------------------------------")
    print(str(classif.GetCorrect()) + " Images correctly classfied ")
    percentage = classif.GetCorrect() / 500
    print(f"amount corrent {percentage}")
    print("--------END---------")


#language translanation
if taskPick == 2:
    
    
    
    translator = translator.Translator()
    translator.LoadAllText("test")
    translator.BuildModel("RNN")
