from clearml import Task, StorageManager, Dataset
import pandas as pd
import numpy as np
import glob
import plotly.express as px


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Step 3 check data",
                 task_type=Task.TaskTypes.data_processing)
args = {
    "dataset_name": "airbnb-nyc-2019-cleaned",
    "dataset_project": "Rental Prices NYC",
    "min_price": 10,
    "max_price": 350
}

task.connect(args)
logger = task.get_logger()
task.execute_remotely()

ds = Dataset.get(dataset_name=args["dataset_name"], dataset_project=args["dataset_project"], dataset_tags=["latest"])
csv_files = glob.glob("%s/*.csv" % ds.get_local_copy())
df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

expected_colums = [
    "id",
    "name",
    "host_id",
    "host_name",
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "room_type",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "last_review",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
]

these_columns = df.columns.values
logger.report_text("Test column names: column names expected: %s",
                   list(expected_colums))
logger.report_text("Test column names: column names in dataset: %s",
                   list(these_columns))
assert list(expected_colums) == list(these_columns)

known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
neigh = set(df['neighbourhood_group'].unique())
logger.report_text("Test column names: neighbourhood group expected: %s",
                   set(known_names))
logger.report_text("Test column names: neighbourhood group in dataset: %s",
                   set(neigh))
assert set(known_names) == set(neigh)

idx = df['longitude'].between(-74.25, - 73.50) & data['latitude'].between(40.5, 41.2)
logger.report_text("Test proper boundaries: unexpected items are %s", np.sum(~idx))
assert np.sum(~idx) == 0

logger.report_text("Test row count: items are %s", df.shape[0])
assert 15000 < df.shape[0] < 1000000

items_ok = df['price'].between(args["min_price"], args["max_price"]).shape[0]
logger.report_text("Price range test, items in range are: %s", items_ok)
assert df.shape[0] == items_ok
