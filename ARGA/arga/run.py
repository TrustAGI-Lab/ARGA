import settings

from clustering import Clustering_Runner
from link_prediction import Link_pred_Runner


dataname = 'citeseer'       # 'cora' or 'citeseer' or 'pubmed'
model = 'arga_vae'          # 'arga_ae' or 'arga_vae'
task = 'clustering'         # 'clustering' or 'link_prediction'

settings = settings.get_settings(dataname, model, task)

if task == 'clustering':
    runner = Clustering_Runner(settings)
else:
    runner = Link_pred_Runner(settings)

runner.erun()

