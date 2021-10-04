testmodel:
	~/bin/python3.8/bin/python3.8 testmodel.py
	~/bin/python3.8/bin/hear-validator hearbaseline.wav2vec2 --model toyzdog.pt
clean:
	rm -rf __pycache__
	rm *~
clean_pt:
	rm -rf pretrained
	rm *.pt
