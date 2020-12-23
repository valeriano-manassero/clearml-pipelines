from clearml import Task, StorageManager


task = Task.init(project_name="iris",
                 task_name="pipeline step 1 dataset artifact",
                 task_type=Task.TaskTypes.data_processing)
task.execute_remotely()

local_iris_pkl = StorageManager.get_local_copy(
    remote_url="https://tableconvert.com/?output=csv&data=https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv")

task.upload_artifact('dataset', artifact_object=local_iris_pkl)

print('uploading artifacts in the background')
print('Done')
