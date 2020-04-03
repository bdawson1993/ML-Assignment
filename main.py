# encoding: utf-8
import classifier
import translator
import concurrent.futures
import multiprocessing
import log
import progressbar


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


    #spawn threads to go through and test the data
    threadCount = multiprocessing.cpu_count()
    classif.testData(0, "cat")
    #with concurrent.futures.ThreadPoolExecutor(max_workers=threadCount) as executor:
     #   for index in range(500):
      #      executor.submit(classif.testData, index, "cat")
            

                    
    log.stop("Testing Finished")

    print(classif.GetCorrect())
    percentage = classif.GetCorrect() / 500
    print(f"amount corrent {percentage}")


#language translanation
if taskPick == 2:
    translator = translator.Translator()
    translator.LoadAllText("test")
    translator.BuildModel("test")







