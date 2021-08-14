from clearml import Task, StorageManager, Dataset


task = Task.init(project_name="mushrooms",
                 task_name="mushrooms step 1 dataset artifact",
                 task_type=Task.TaskTypes.data_processing)
args = {
    "dataset_name": "DATASET_NAME",
    "dataset_s3_path": "DATASET_S3_PATH"
}
task.connect(args)
logger = task.get_logger()
task.execute_remotely()

logger.report_text("Downloading mushrooms dataset")
local_mushrooms_dataset = StorageManager.get_local_copy(
    remote_url="https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv")

ds = Dataset.create(dataset_name=args["dataset_name"], use_current_task=True)
ds.add_files(local_mushrooms_dataset, recursive=True)
logger.report_text("S3 upload -> mushrooms dataset")
ds.upload(output_url=args["dataset_s3_path"])
ds.finalize()
ds.tags = []
ds.tags = ['latest']
