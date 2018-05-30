import unittest
from srf.task import TaskFactory
from srf.task import Task, TaskReconstruction, TaskTest
from srf.config import clear_config, update_config


class TestTaskFactoryCustomTest(unittest.TestCase):
    def test_make_recon_task(self):
        config = {}  # REAL config


class TestTaskFactory(unittest.TestCase):
    def setUp(self):
        clear_config()

    def tearDown(self):
        clear_config()

    def set_test_task_configs(self):
        update_config({'task_type': 'TaskTest',
                       'task_name': ' test'})

    def assertTestTaskCorrectlyInitilized(self, task):
        from srf.task import TaskTest
        self.assertTrue(isinstance(task, TaskTest), 'Invalid task type')
        self.assertEqual(task.info.name, 'test', 'Wrong task name')

    def test_make_task(self):
        self.set_make_test_task_configs()
        result = TaskFactory.make_task()
        self.assertTestTaskCorrectlyInitilized(result)

    def test_get_task_type(self):
        task_type_cls_map = {
            'TaskReconstruction': TaskReconstruction,
            'TaskTest': TaskTest
        }
        for k, v in task_type_cls_map:
            update_config({'task_type': k})
            result_cls = TaskFactory.get_task_type()
            self.assertEqual(result_cls, v, "Invalid task type for {}.".format(k))
