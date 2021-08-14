from clearml import Task
from clearml.automation.controller import PipelineController


task = Task.init(project_name='mushrooms', task_name='Model creation mushrooms',
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)
args = {
    'worker_queue': 'default',
    'dataset_s3_path': 's3://minio-hl.minio:9000/clearml/data',
    "dataset_name": "mushrooms_dataset",
}
task.connect(args)
logger = task.get_logger()
task.execute_remotely()

logger.report_text(args["dataset_s3_path"])

pipe = PipelineController(default_execution_queue='default',
                          add_pipeline_tags=False)

pipe.add_step(name='stage_data',
              base_task_project='mushrooms',
              base_task_name='mushrooms step 1 dataset artifact',
              parameter_override={
                  "General/dataset_s3_path": "s3://minio-hl.minio:9000/clearml/data",
                  "General/dataset_name": "mushrooms_dataset"
                  },
              execution_queue=args["worker_queue"])

pipe.add_step(name='stage_train',
              parents=['stage_data', ],
              base_task_project='mushrooms',
              base_task_name='mushrooms step 2 train model',
              parameter_override={
                  "General/dataset_name": args["dataset_name"]
                  },
              execution_queue=args["worker_queue"])

pipe.start()
pipe.wait()
pipe.stop()
