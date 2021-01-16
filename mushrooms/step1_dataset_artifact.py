from clearml import Task, StorageManager


task = Task.init(project_name="mushrooms",
                 task_name="mushrooms step 1 dataset artifact",
                 task_type=Task.TaskTypes.data_processing)
task.execute_remotely()

local_mushrooms_dataset = StorageManager.get_local_copy(
    remote_url="https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv")
task.upload_artifact("dataset", artifact_object=local_mushrooms_dataset)

print('uploading csv dataset in the background')
