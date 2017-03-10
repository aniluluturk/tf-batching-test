# mnist tf batching test #

Sample files for separated batch creation, training and testing of MNIST dataset, using tensorflow


```python
python mnist_create_batch.py -n 5 -s 100 -r  # (create batch of 5 files, each contining 
                                             # 100 training examples, and randomize them (-r))
python mnist_train_save.py -s 0 -n 3 -d 1    # train device/instance 1, using examples between
                                             # 0 and 0+3 
python mnist_test_load.py -d 1               # load checkpoint model for device/instance 1,
                                             # and calculate accuracy
```
