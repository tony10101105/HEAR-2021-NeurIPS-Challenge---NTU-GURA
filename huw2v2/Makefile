testmodel:
	~/bin/python3.8/bin/python3.8 testmodel.py
clean:
	rm -rf __pycache__
	rm *~
clean_pt:
	rm -rf pretrained
	rm *.pt
compute:
	CUDA_VISIBLE_DEVICES=2 \
	time ~/bin/python3.8/bin/python3.8 -m heareval.embeddings.runner huw2v2 \
					--model huw2v2.pt \
					--tasks-dir /tmp2/b08902126/hear-2021.0.3/tasks/ &> 0.out &
eval:
	CUDA_VISIBLE_DEVICES=2 ~/bin/python3.8/bin/python3.8 -m heareval.predictions.runner ./embeddings/huw2v2/* &> 0.out &
