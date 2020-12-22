import os
from trains import Task, StorageManager
from kaggle.api.kaggle_api_extended import KaggleApi
import tarfile

task = Task.init(project_name="inat-2019", task_name="pipeline step 1 dataset artifact")
task.execute_remotely()

args = {
    "kaggle_username": "",
    "kaggle_key": "",
}
task.connect(args)

os.environ["KAGGLE_USERNAME"] = args["kaggle_username"]
os.environ["KAGGLE_KEY"] = args["kaggle_key"]
api = KaggleApi()
api.authenticate()
archive_file = "inat-2019.tar.gz"
api.competition_download_file("inaturalist-2019-fgvc6", file_name=archive_file)
tar = tarfile.open(archive_file, "r:gz")
tar.extractall("./inat-2019")
tar.close()

task.upload_artifact("dataset", artifact_object="./inat-2019")
