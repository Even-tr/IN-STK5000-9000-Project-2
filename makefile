
all:
	rm output.txt
	python script.py --infile='diabetes.csv' >> output.txt
	cat output.txt
run:
	rm output.txt
ifdef file
	python script.py --infile=file >> output.txt
else
	python script.py --infile='diabetes.csv' >> output.txt
endif
	cat output.txt

normal:
	python ./pipeline.py diabetes.csv
	
anon:
	python ./privacyPipe.py diabetes.csv anon.csv 0.95
	python ./pipeline.py anon.csv


analysis:
	make normal
	make anon