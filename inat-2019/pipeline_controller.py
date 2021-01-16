from clearml import Task
from clearml.automation.controller import PipelineController


task = Task.init(project_name="inat-2019", task_name="Model creation inat-2019",
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)
args = {
    "worker_queue": "default",
}
task.connect(args)

pipe = PipelineController(default_execution_queue="default",
                          add_pipeline_tags=False)

pipe.add_step(name="stage_data",
              base_task_project="inat-2019",
              base_task_name="inat-2019 step 1 dataset artifact",
              execution_queue=args["worker_queue"])

pipe.start()
pipe.wait()
pipe.stop()
