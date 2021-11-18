.PHONY: prepare-pip prepare-pacman-nvidia all destroy clear

prepare-pip:
	pip install numpy pandas matplotlib seaborn sklearn tensorflow

prepare-pacman-nvidia:
	sudo pacman -S --needed python cuda cudnn

env:
	python -m venv env

build:
	mkdir build

build/filtered_train.csv: src/train.csv build scripts/filter_data.py
	python scripts/filter_data.py

build/network_accuracies.csv: build/filtered_train.csv scripts/train_all_networks.py
	python scripts/train_all_networks.py

build/best_answers.csv: build/network_accuracies.csv scripts/test_best_network.py
	python scripts/test_best_network.py

build/train_graphics: build/network_accuracies.csv scripts/draw_train_graphics.py # здесь ожидаю ошибку с зависимостями, т.к. вряд ли проверяется изменение содержимого папки
	mkdir -p build/train_graphics
	rm -f build/train_graphics/*.png
	python scripts/draw_train_graphics.py

build/report.md: build/train_graphics build/best_answers.csv scripts/compile_report.py src/report_template.md
	python scripts/compile_report.py

#build/report.pdf: build/report.md src/report_header.pdf
# WIP

clear:
	rm -rf build

destroy: clear
	rm -rf env

all: build/report.md
