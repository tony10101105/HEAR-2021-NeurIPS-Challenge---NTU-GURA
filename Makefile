testmodel:
	~/bin/python3.8/bin/python3.8 testmodel.py
	~/bin/python3.8/bin/hear-validator hearbaseline.wav2vec2 --model hubert-large-ll60k.pt
clean:
	rm -rf __pycache__
	rm *~
clean_pt:
	rm -rf pretrained
	rm *.pt
compute:
	CUDA_VISIBLE_DEVICES=0 \
	time ~/bin/python3.8/bin/python3.8 -m heareval.embeddings.runner hubert_large_ll60k \
					--model hubert-large-ll60k.pt \
					--tasks-dir /tmp2/b08902126/hear-2021.0.3/tasks/
eval:
	CUDA_VISIBLE_DEVICES=0,2,5 ~/bin/python3.8/bin/python3.8 -m heareval.predictions.runner ./embeddings/hubert_large_ll60k/*
