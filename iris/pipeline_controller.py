from trains import Task
from trains.automation.controller import PipelineController


# Connecting trains with the current process,
# from here on everything is logged automatically
task = Task.init(project_name='examples', task_name='iris demo',
                 task_type=Task.TaskTypes.controller, reuse_last_task_id=False)

pipe = PipelineController(default_execution_queue='default', add_pipeline_tags=False)
pipe.add_step(name='stage_data', base_task_project='examples', base_task_name='iris step 1 dataset artifact')
pipe.add_step(name='stage_process', parents=['stage_data', ],
              base_task_project='examples', base_task_name='iris step 2 process dataset',
              parameter_override={'General/dataset_url': '${stage_data.artifacts.dataset.url}',
                                  'General/test_size': 0.25})
pipe.add_step(name='stage_train', parents=['stage_process', ],
              base_task_project='examples', base_task_name='iris step 3 train model',
              parameter_override={'General/dataset_task_id': '${stage_process.id}'})

# Starting the iris (in the background)
pipe.start()
# Wait until iris terminates
pipe.wait()
# cleanup everything
pipe.stop()

print('done')
