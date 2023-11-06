
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
	python ./pipe.py diabetes.csv
	
anon:
	python ./privacy.py diabetes.cvs anon.cvs
	python ./pipe.py anon.csv


analysis:
	make normal
	make anon