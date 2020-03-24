import classifier
import concurrent.futures
import multiprocessing
import log


classif = classifier.classifier()
print("KNN")
print("GNB")
print("SVM")
classTag = input("Please Enter Classifier to use:")
classif.BuildModel(classTag)


#spawn threads to go through and test the data
threadCount = multiprocessing.cpu_count()
log.start("Testing Data")
with concurrent.futures.ThreadPoolExecutor(max_workers=threadCount) as executor:
       for index in range(500):
           executor.submit(classif.testData, index)
log.stop("Testing Finished")

print(classif.GetCorrect())
percentage = classif.GetCorrect() / 500
print(f"amount corrent {percentage}")





