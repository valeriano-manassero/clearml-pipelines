from trains import Task
from trains.automation.controller import PipelineController


task = Task.init(project_name='inat-2019', task_name='Model creation pipeline',
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)
task.execute_remotely()

pipe = PipelineController(default_execution_queue='default',
                          add_pipeline_tags=False)

pipe.add_step(name='stage_data', base_task_project='inat-2019', base_task_name='pipeline step 1 dataset artifact')

pipe.start()
pipe.wait()
pipe.stop()
