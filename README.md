# Abstractive-Summarization-Dataset

This repository is heavily borrowed from https://github.com/mjc92/GetToThePoint 
My work on this, has just started. Easiest way to obtain data for Abstractive summarization is from https://github.com/abisee/pointer-generator
but the final binary files are Protobuf binary files that we get in it, you can directly use them in pytorch by just decoding using Tensorflow's example_pb2
but I find this method by converting whole dataset into a pickle file and loading it simmpler as it is not a larger dataset (it's just around 1.3 GB for training)
scripts for transporting protobuf to pickle are available at finished_files directory. Enjoy exploring summarization tasks.
