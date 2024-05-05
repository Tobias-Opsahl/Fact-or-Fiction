# Names of files, for copying and taring
SOURCES := README.md Makefile
TAR_FILE_NAME := home_exam.tar.gz
FOX_PATH := ec-tobiasao@fox.educloud.no

tar_files:
	tar -czvf $(TAR_FILE_NAME) $(SOURCES)

fox_login:
	ssh $(FOX_PATH)

ificp:
	scp -v -r $(TAR_FILE_NAME) $(FOX_PATH):~/home-exam/

tar_and_ificp: tar_files ificp

detar:
	tar -xzvf $(TAR_FILE_NAME)

# Remote to local
# scp ec-tobiasao@fox.educloud.no:/fp/projects01/ec30/corpora/enwiki/xab  ./data/

# Decompress
# tar -xzvf home_exam.tar.gz