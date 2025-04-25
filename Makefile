SRCDIR = src
PKGS = pandas numpy

GREY   = \033[30m
GREEN  = \033[32m
YELLOW = \033[33m
RESET  = \033[0m\n

.SILENT:

all: check
	printf "$(GREEN) Packages Ready! $(RESET)"
	printf "$(GREY)  Usage:$(YELLOW) make gen, a $(RESET)"

check:
	for pkg in $(PKGS); do \
		if ! python3 -c "import $$pkg" 2>/dev/null; then \
			pip3 install $$pkg > /dev/null 2>&1; \
		fi; \
	done

a:
	python3 $(SRCDIR)/main.py data/data.csv

gen:
	python3 $(SRCDIR)/split.py data/data.csv

clean:
	# rm -rf

fclean: clean
	# rm -rf 

gpush: fclean
	git add .
	git commit -m "start"
	git push

re: fclean all
