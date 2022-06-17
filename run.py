import csv
import os
import resource
import sys
import time

import deepmatcher as dm
import fcntl

path = sys.argv[1]
dataset_name = sys.argv[2]

train, validation, test = dm.data.process(path=path,
                                          train='train.csv',
                                          validation='valid.csv',
                                          test='test.csv')
model = dm.MatchingModel()

# Training
start_time = time.time()
model.run_train(train, validation, epochs=15,
                best_save_path=path + 'best_model.pth')
train_time = time.time() - start_time
train_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Testing
start_time = time.time()
stats = model.run_eval(test, return_stats=True)
test_time = time.time() - start_time
test_max_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

# Persist results
result_file = '/home/remote/u6852937/projects/results.csv'
file_exists = os.path.isfile(result_file)

with open(result_file, 'a') as results_file:
  heading_list = ['method', 'dataset_name', 'train_time', 'test_time',
                  'train_max_mem', 'test_max_mem', 'TP', 'FP', 'FN',
                  'TN', 'Pre', 'Re', 'F1', 'Fstar']
  writer = csv.DictWriter(results_file, fieldnames=heading_list)

  if not file_exists:
    writer.writeheader()

  p = stats.precision() / 100.0
  r = stats.recall() / 100.0
  f_star = 0 if (p + r - p * r) == 0 else p * r / (p + r - p * r)
  fcntl.flock(results_file, fcntl.LOCK_EX)
  result_dict = {
    'method': 'deepmatcher',
    'dataset_name': dataset_name,
    'train_time': round(train_time, 2),
    'test_time': round(test_time, 2),
    'train_max_mem': train_max_mem,
    'test_max_mem': test_max_mem,
    'TP': stats.tps.item(),
    'FP': stats.fps.item(),
    'FN': stats.fns.item(),
    'TN': stats.tns.item(),
    'Pre': ('{prec:.2f}').format(prec=stats.precision()),
    'Re': ('{rec:.2f}').format(rec=stats.recall()),
    'F1': ('{f1:.2f}').format(f1=stats.f1()),
    'Fstar': ('{fstar:.2f}').format(fstar=f_star * 100)
  }
  writer.writerow(result_dict)
  fcntl.flock(results_file, fcntl.LOCK_UN)
