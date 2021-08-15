from clearml import Task, StorageManager, Dataset


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Step 1 create dataset",
                 task_type=Task.TaskTypes.data_processing)
args = {
}
task.connect(args)
logger = task.get_logger()
task.execute_remotely()

logger.report_text("Downloading AirBnb NYC 2019 dataset dataset")
dataset = StorageManager.get_local_copy(
    remote_url="https://raw.githubusercontent.com/adishourya/Airbnb/master/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
    
ds = Dataset.create(dataset_name="airbnb-nyc-2019", use_current_task=True)
ds.add_files(dataset, recursive=True)
logger.report_text("S3 upload -> dataset")
ds.upload()
ds.finalize()
ds.tags = []
ds.tags = ['latest']