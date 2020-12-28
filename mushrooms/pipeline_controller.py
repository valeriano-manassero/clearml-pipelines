from trains import Task
from trains.automation.controller import PipelineController


task = Task.init(project_name='mushrooms', task_name='Model creation mushrooms',
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)
args = {
    'worker_queue': 'default',
}
task.connect(args)
task.execute_remotely()

pipe = PipelineController(default_execution_queue='default',
                          add_pipeline_tags=False)

pipe.add_step(name='stage_data',
              base_task_project='mushrooms',
              base_task_name='mushrooms step 1 dataset artifact',
              execution_queue=args["worker_queue"])
pipe.add_step(name='stage_train',
              parents=['stage_data', ],
              base_task_project='mushrooms',
              base_task_name='mushrooms step 2 train model',
              parameter_override={'General/stage_data_task_id': '${stage_data.id}'},
              execution_queue=args["worker_queue"])

pipe.start()
pipe.wait()
pipe.stop()
