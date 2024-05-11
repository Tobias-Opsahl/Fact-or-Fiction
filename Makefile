# Names of files, for copying and taring
SOURCES := README.md Makefile constants.py utils.py retrieve_subgraphs.py scripts/run_retrieve_subgraph.slurm
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
# scp ec-tobiasao@fox.educloud.no:/fp/projects01/ec30/factkg/dbpedia/dbpedia_2015_undirected_light.pickle ./home-exam/data/dbpedia/
# scp -r ec-tobiasao@fox.educloud.no:/fp/projects01/ec30/factkg/simple/ ./home-exam/data/simple/
# scp -r ec-tobiasao@fox.educloud.no:/fp/projects01/ec30/factkg/full/ ./home-exam/data/full/

# scp -r ec-tobiasao@fox.educloud.no:~/home-exam/data/full_val_1.csv ./data/full_val_1.csv

# Run script
# sbatch scripts/...
# squeue -u ec-tobiasao --start
# sinfo
