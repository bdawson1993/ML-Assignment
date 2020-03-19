import classifier
import concurrent.futures
import multiprocessing
import log

classif = classifier.classifier()
threadCount = multiprocessing.cpu_count()

#spawn threads to go through
log.start("Testing Data")
with concurrent.futures.ThreadPoolExecutor(max_workers=threadCount) as executor:
       for index in range(500):
           executor.submit(classif.testData, index)
log.stop("Testing Finished")

print(classif.GetCorrect())
percentage = classif.GetCorrect() / 500
print(f"amount corrent {percentage}")





