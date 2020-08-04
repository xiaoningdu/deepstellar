

## Prepare Environment
python=3.6

pip install -r requirement.txt


## To prepare an RNN model

python -m mnist_demo.mnist_lstm


## Generate the DTMC abstract model

python abstraction_runner.py -dl_model test/rnn_model/model.h5  -profile_save_path test/output/profile_save -comp_num 128 -k 3 -m 10 -bits 8 -name_prefix lstm_mnist -abst_save_path test/output/abst_model


## Coverage Guided Testing

python -m fuzzer.ConstructInitialSeeds -dl_model test/rnn_model/model.h5  -output_path ../fuzz_data/initialseeds

python -m fuzzer.image_fuzzer -i ../fuzz_data/initialseeds -o ../fuzz_data/fuzzing-out-1/lstm-trans-3-10 -model_type mnist -dl_model test/rnn_model/model.h5 -criteria state -pkl_path test/output/abst_model/wrapper_lstm_mnist_3_10.pkl


## Evaluation of the testing

### Check the coverage metrics of the fuzzing output queue:

python -m evaluation_scripts.fuzzing.coverage_analyzer -dl_model test/rnn_model/model.h5 -wrapper test/output/abst_model/wrapper_lstm_mnist_3_10.pkl -inputs_folder ../fuzz_data/fuzzing-out-1/lstm-trans-3-10/queue -type queue

### Check the coverage metrics of the initial seeds:

python -m evaluation_scripts.fuzzing.coverage_analyzer -dl_model test/rnn_model/model.h5 -wrapper test/output/abst_model/wrapper_lstm_mnist_3_10.pkl -inputs_folder ../fuzz_data/initialseeds -type seeds

### Check the number of unique crashes

python -m evaluation_scripts.fuzzing.check_unique_crash -i ../fuzz_data/fuzzing-out-1/lstm-trans-3-10/crashes