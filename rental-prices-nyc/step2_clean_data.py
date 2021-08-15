from clearml import Task, StorageManager, Dataset
import pandas as pd
import glob


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Step 1 clean data",
                 task_type=Task.TaskTypes.data_processing)
args = {
}
task.connect(args)
logger = task.get_logger()
task.execute_remotely()

logger.report_text("Downloading AirBnb NYC 2019 dataset dataset")


ds = Dataset.get(dataset_name="airbnb-nyc-2019", dataset_project="Rental Prices NYC", dataset_tags=["latest"])
csv_files = glob.glob("%s/*.csv" % ds.get_local_copy())
dataframe = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

min_price = args.min_price
max_price = args.max_price
idx = dataframe['price'].between(min_price, max_price)
dataframe = dataframe[idx].copy()
logger.report_text("Dataset price outliers removal outside range: %s-%s", args.min_price, args.max_price)
dataframe['last_review'] = pd.to_datetime(dataframe['last_review'])
logger.report_text("Dataset last_review data type fix")

idx = dataframe['longitude'].between(-74.25, -73.50) & dataframe['latitude'].between(40.5, 41.2)
dataframe = dataframe[idx].copy()

dataframe.to_csv("datset.csv", index=False)

ds = Dataset.create(dataset_name="airbnb-nyc-2019-cleaned", parent_datasets=["airbnb-nyc-2019"], use_current_task=True)
ds.add_files("dataset.csv")
logger.report_text("S3 upload -> dataset")
ds.upload()
ds.finalize()
ds.tags = []
ds.tags = ['latest']
