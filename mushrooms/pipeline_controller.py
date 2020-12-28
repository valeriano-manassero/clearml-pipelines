from trains import Task
from trains.automation.controller import PipelineController


task = Task.init(project_name='mushrooms', task_name='Model creation pipeline',
                 task_type=Task.TaskTypes.controller,
                 reuse_last_task_id=False)

pipe = PipelineController(default_execution_queue='default',
                          add_pipeline_tags=False)

pipe.add_step(name='stage_data',
              base_task_project='mushrooms',
              base_task_name='pipeline step 1 dataset artifact')
pipe.add_step(name='stage_train',
              parents=['stage_data', ],
              base_task_project='mushrooms',
              base_task_name='pipeline step 2 train model',
              parameter_override={'General/stage_data_task_id': '${stage_data.id}'})

pipe.start()
pipe.wait()
pipe.stop()
