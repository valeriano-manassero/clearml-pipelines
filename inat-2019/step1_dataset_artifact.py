import os
import shutil
from trains import Task
import tarfile


task = Task.init(project_name="inat-2019",
                 task_name="inat-2019 step 1 dataset artifact",
                 task_type=Task.TaskTypes.data_processing)
args = {
    "kaggle_username": "kaggle_username",
    "kaggle_key": "kaggle_key",
}
task.connect(args)
task.execute_remotely()

os.environ["KAGGLE_USERNAME"] = args["kaggle_username"]
os.environ["KAGGLE_KEY"] = args["kaggle_key"]
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
archive_file = "train_val2019.tar.gz"
tmp_folder = "./inat-2019"
api.competition_download_file("inaturalist-2019-fgvc6", archive_file)
tar = tarfile.open(archive_file, "r:gz")
tar.extractall(tmp_folder)
tar.close()

task.upload_artifact("dataset", artifact_object=tmp_folder)

shutil.rmtree(tmp_folder)
os.remove(archive_file)
