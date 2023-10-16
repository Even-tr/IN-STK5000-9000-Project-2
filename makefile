
all:
	rm output.txt
	python script.py diabetes.csv >> output.txt
	cat output.txt

run:
	rm output.txt
ifdef file
	python script.py file >> output.txt
else
	python script.py diabetes.csv >> output.txt
endif
	cat output.txt