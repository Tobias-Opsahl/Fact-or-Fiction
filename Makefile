# Names of files, for copying and taring
SOURCES := test_baseline.py evaluate.py README.md Makefile constants.py utils.py retrieve_subgraphs.py models.py train.py run_stuff.py make_subgraph_embeddings.py datasets.py baseline/baseline_train.py baseline/kg.py scripts/*.slurm
TAR_FILE_NAME := home_exam.tar.gz
FOX_PATH := ec-tobiasao@fox.educloud.no

tar_files:
	tar -czvf $(TAR_FILE_NAME) $(SOURCES)

fox_login:
	ssh $(FOX_PATH)

ificp:
	scp -v -r $(TAR_FILE_NAME) $(FOX_PATH):~/home-exam/

mlcp:
	scp -v -r -J tobiasao@login.uio.no $(TAR_FILE_NAME) tobiasao@ml1.hpc.uio.no:TobiasaoThesis/in5550_home_exam/

tar_and_ificp: tar_files ificp

detar:
	tar -xzvf $(TAR_FILE_NAME)

