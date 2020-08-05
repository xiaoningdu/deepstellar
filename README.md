# DeepStellar: Model-Based Quantitative Analysis of Stateful Deep Learning Systems

## Prepare Environment
python=3.6

pip install -r requirement.txt


## To prepare an RNN model

```shell script
python -m mnist_demo.mnist_lstm
```


## Generate the DTMC abstract model

```shell script
python abstraction_runner.py -dl_model test/rnn_model/model.h5  -profile_save_path test/output/profile_save -comp_num 128 -k 3 -m 10 -bits 8 -name_prefix lstm_mnist -abst_save_path test/output/abst_model

```


## Coverage Guided Testing

### Construct initial seeds
```shell script
python -m fuzzer.construct_initial_seeds -dl_model test/rnn_model/model.h5  -output_path ../fuzz_data/initialseeds
```

### Launch the testing process
```shell script
python -m fuzzer.image_fuzzer -i ../fuzz_data/initialseeds -o ../fuzz_data/fuzzing-out-1/lstm-trans-3-10 -model_type mnist -dl_model test/rnn_model/model.h5 -criteria state -pkl_path test/output/abst_model/wrapper_lstm_mnist_3_10.pkl
```


## Evaluation of the testing

### Check the coverage metrics of the fuzzing output queue:

```shell script
python -m evaluation_scripts.fuzzing.coverage_analyzer -dl_model test/rnn_model/model.h5 -wrapper test/output/abst_model/wrapper_lstm_mnist_3_10.pkl -inputs_folder ../fuzz_data/fuzzing-out-1/lstm-trans-3-10/queue -type queue
```

### Check the coverage metrics of the initial seeds:

```shell script
python -m evaluation_scripts.fuzzing.coverage_analyzer -dl_model test/rnn_model/model.h5 -wrapper test/output/abst_model/wrapper_lstm_mnist_3_10.pkl -inputs_folder ../fuzz_data/initialseeds -type seeds
```

### Check the number of unique crashes

```shell script
python -m evaluation_scripts.fuzzing.check_unique_crash -i ../fuzz_data/fuzzing-out-1/lstm-trans-3-10/crashes
```

### If you would like to use Deepsteller in your research, please cite our FSE'19 paper:

```shell script
@inproceedings{10.1145/3338906.3338954,
author = {Du, Xiaoning and Xie, Xiaofei and Li, Yi and Ma, Lei and Liu, Yang and Zhao, Jianjun},
title = {DeepStellar: Model-Based Quantitative Analysis of Stateful Deep Learning Systems},
year = {2019},
booktitle = {Proceedings of the 2019 27th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
pages = {477–487},
series = {ESEC/FSE 2019}
}
```

  

