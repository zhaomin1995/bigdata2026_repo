# Duration

Data: 
`wget https://zhaomin.s3.us-east-2.amazonaws.com/public_projects/duration_final_data.zip`

Environment:

```
mkdir envs
cd envs
python3 -m venv loc_tmp3
source loc_tmp3/bin/activate
cd ..
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip3 install -U scikit-learn
sudo yum install tmux
unzip duration_final_data.zip
```

Run the experiments:


## Test the response

We've saved our LLM and VLM respoonses under experimental_results, you can directly test them. 

```
python response_test.py
```

### Reproduce the response

```
# For Gpt-5-mini and grok4, you need a openkey subscription
python llm_openai.py
# For Gemini and Claude series, you need a google cloud vetex AI subscription
llm_gemini.py
# For LLama3
python llm_llama3.py
# For Qwen series
python llm_qwen.py
# For GLM
python llm_glm.py
# For PLTM
python main.py -model no_matcher -num_epochs 10 -batch_size 8 -learning_rate 1e-06 -early_stopping 2 -use_image -pooling maxpooling
python main.py --repeated_run --batch_size 64 --learning_rate 1e-06 --num_epochs 1000 --early_stopping 5 --model clip2_TIL
```
