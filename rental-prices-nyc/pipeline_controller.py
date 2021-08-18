from clearml import Task
from clearml.automation.controller import PipelineController


task = Task.init(project_name="Rental Prices NYC",
                 task_name="Pipeline controller",
                 task_type=Task.TaskTypes.controller
                 )

args = {
    "dataset_name_raw": "airbnb-nyc-2019-raw",
    "dataset_name_cleaned": "airbnb-nyc-2019-cleaned",
    "dataset_project": "Rental Prices NYC",
    "min_price": 10,
    "max_price": 350,
    "worker_queue": "default"
}

task.connect(args)
logger = task.get_logger()
task.execute_remotely()

pc = PipelineController(default_execution_queue=args["worker_queue"],
                          add_pipeline_tags=False)

project_name = "Rental Prices NYC"
pc.add_step(name='create_dataset',
            base_task_project=project_name,
            base_task_name="Step 1 create dataset",
            execution_queue=args["worker_queue"])
pc.add_step(name='clean_data',
            parents=['create_dataset', ],
            base_task_project=project_name,
            base_task_name="Step 2 clean data",
            execution_queue=args["worker_queue"])

logger.report_text("Pipeline started")
pc.start()
logger.report_text("pipeline in progress")
pc.wait()
logger.report_text("Pipeine completed")
pc.stop()
