from clearml import Task, StorageManager, Dataset
import pandas as pd
import glob
import plotly.express as px


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Step 2 clean data",
                 task_type=Task.TaskTypes.data_processing)
args = {
    "dataset_name_raw": "airbnb-nyc-2019-raw",
    "dataset_name_cleaned": "airbnb-nyc-2019-cleaned",
    "dataset_project": "Rental Prices NYC",
    "min_price": 10,
    "max_price": 350  
}

dataset_cleaned_name = "dataset.csv"

task.connect(args)
logger = task.get_logger()
task.execute_remotely()

ds_raw = Dataset.get(dataset_name=args["dataset_name_raw"], dataset_project=args["dataset_project"], dataset_tags=["latest"])
csv_files = glob.glob("%s/*.csv" % ds_raw.get_local_copy())
df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

min_price = args["min_price"]
max_price = args["max_price"]
idx = df['price'].between(min_price, max_price)
df = df[idx].copy()
logger.report_text("Dataset price outliers removal outside range: %s-%s", args["min_price"], args["max_price"])
df['last_review'] = pd.to_datetime(df['last_review'])
logger.report_text("Dataset last_review data type fix")

idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
df = df[idx].copy()

df.to_csv(dataset_cleaned_name, index=False)

try:
    original_ds = Dataset.get(dataset_name=args["dataset_name_cleaned"], dataset_project=args["dataset_project"], dataset_tags=["latest"])
    logger.report_text("Dataset %s already exists, creating the child one" % args["dataset_name_cleaned"])
    ds = Dataset.create(dataset_name=args["dataset_name_cleaned"], parent_datasets=[original_ds.id], dataset_project=args["dataset_project"])
except ValueError:
    original_ds = None
    logger.report_text("Dataset %s does not exists, creating the first one" % args["dataset_name_cleaned"])
    ds = Dataset.create(dataset_name=args["dataset_name_cleaned"], dataset_project=args["dataset_project"])
ds.add_files(dataset_cleaned_name )
ds.upload()
ds.finalize()
ds.tags = []
ds.tags = ['latest']
if original_ds:
    original_ds.tags = []

fig = px.histogram(df, x="price", marginal="box")
logger.report_plotly(title="TOTAL Price distribution", series="Price", iteration=0, figure=fig)

fig = px.histogram(df, x="price", marginal="box", color="neighbourhood_group")
logger.report_plotly(title="Price distribution by Neighbourhood", series="Price", iteration=0, figure=fig)
