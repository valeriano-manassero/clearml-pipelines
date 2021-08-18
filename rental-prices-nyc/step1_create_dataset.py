from clearml import Task, StorageManager, Dataset
import pandas as pd
import plotly.express as px


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Step 1 create dataset",
                 task_type=Task.TaskTypes.data_processing)
args = {
}
task.connect(args)
logger = task.get_logger()
task.execute_remotely()

dataset_name="airbnb-nyc-2019-raw"
dataset_project="Rental Prices NYC"
try:
    logger.report_text("Dataset %s already exists, creating the child one" % dataset_name)
    original_ds = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project, dataset_tags=["latest"])
    ds = Dataset.create(dataset_name=dataset_name, parent_datasets=[original_ds.id], dataset_project=dataset_project)
except ValueError:
    logger.report_text("Dataset %s does not exists, creating the first one" % dataset_name)
    ds = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project)

logger.report_text("Downloading AirBnb NYC 2019 dataset")
dataset_csv = StorageManager.get_local_copy(
    remote_url="https://raw.githubusercontent.com/adishourya/Airbnb/master/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
    
ds.add_files(dataset_csv)
logger.report_text(ds.list_files())
ds.upload()
ds.finalize()
ds.tags = []
ds.tags = ['latest']
if original_ds:
    original_ds.tags = []
    original_ds

df = pd.read_csv(dataset_csv)
fig = px.histogram(df, x="price", marginal="box")
logger.report_plotly(title="TOTAL Price distribution", series="Price", iteration=0, figure=fig)

fig = px.histogram(df, x="price", marginal="box", color="neighbourhood_group")
logger.report_plotly(title="Price distribution by Neighbourhood", series="Price", iteration=0, figure=fig)
