import os
import glob
import pickle
import pandas as pd
import numpy as np

from dask.diagnostics import ProgressBar

from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

from pyscenic.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell

import seaborn as sns


def run():

	DATA_FOLDER="tmp"
	RESOURCES_FOLDER="resources/"
	DATABASE_FOLDER = "databases/"
	SCHEDULER="123.122.8.24:8786"
	DATABASES_GLOB = os.path.join(DATABASE_FOLDER, "mm10__refseq-r80__500bp_up_and_100bp_down_tss.mc9nr.feather")
	MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDER, "motifs-v9-nr.mgi-m0.001-o0.0.tbl")
	MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'mm_tfs.txt')
	SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, "GSE60361_C1-3005-Expression.txt")
	REGULONS_FNAME = os.path.join(DATA_FOLDER, "regulons.p")
	MOTIFS_FNAME = os.path.join(DATA_FOLDER, "motifs.csv")


	ex_matrix = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0).T
	print(ex_matrix.shape)


	tf_names = load_tf_names(MM_TFS_FNAME)


	db_fnames = glob.glob(DATABASES_GLOB)
	print(db_fnames)

	def name(fname):
	    return(os.path.basename(fname).split(".")[0])
	dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
	print(dbs)


	adjacencies = grnboost2(ex_matrix, tf_names=tf_names, verbose=True)

	modules = list(modules_from_adjacencies(adjacencies, ex_matrix))

	# Calculate a list of enriched motifs and the corresponding target genes for all modules.
	with ProgressBar():
	    df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME)

	# Create regulons from this table of enriched motifs.
	regulons = df2regulons(df)

	# Save the enriched motifs and the discovered regulons to disk.
	df.to_csv(MOTIFS_FNAME)
	with open(REGULONS_FNAME, "wb") as f:
	    pickle.dump(regulons, f)


	df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME, client_or_address='dask_multiprocessing')

	df = load_motifs(MOTIFS_FNAME)
	with open(REGULONS_FNAME, "rb") as f:
	    regulons = pickle.load(f)


	auc_mtx = aucell(ex_matrix, regulons, num_workers=4)
	sns_plot = sns.clustermap(auc_mtx, figsize=(8,8))

	sns_plot.savefig("output.png")

if __name__ == "__main__":
   run()



