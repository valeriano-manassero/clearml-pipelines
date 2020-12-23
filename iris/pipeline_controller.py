from trains import Task
from trains.automation.controller import PipelineController


task = Task.init(project_name='iris', task_name='Model creation pipeline',
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)


pipe = PipelineController(default_execution_queue='default',
                          add_pipeline_tags=False)

pipe.add_step(name='stage_data', base_task_project='iris', base_task_name='pipeline step 1 dataset artifact')

pipe.start()
pipe.wait()
pipe.stop()
