 .PHONY: 

train:
	@python train.py

install:
	@pip install -r requirements.txt

lint:
	@flake8

eval:
	@python evaluate.py

plots:
	@python plots.py

visualize_outputs:
	@python visualize_outputs.py

submit:
	@qsub qsub.pbs

status:
	@qstat
