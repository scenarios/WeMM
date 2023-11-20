export LC_ALL="en_US.UTF-8"
pip install scikit-learn

#export CUDA_VISIBLE_DEVICES=7
dtype=bf16
model_path=./model_zoo/hf_models/
results_dir=./tmp/
# get model answers for yes or no
/usr/local/python/bin/python3 ./mme_eval.py \
     --output_dir ${results_dir} \
     --model-path ${model_path} \
     --beam_search \
     --dtype ${dtype} \
     --benchmark ./MME_Benchmark_release_version
