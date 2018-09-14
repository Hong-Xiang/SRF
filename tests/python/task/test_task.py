import unittest
from srf.task import Task
from dxl.learn import Graph


class TestTask(unittest.TestCase):
    def add_simple_config(self):
        from srf.config import update_config
        task_name = 'simple_task'
        update_config('simple_task', {'key': 'value'})
        return task_name

    def make_test_task(self):
        from srf.task import TaskTest
        return TaskTest('test')

    def assertConfigCorrectlyLoaded(self, task):
        self.assertEqual(task.config('key'), 'value')

    def assertRunCalled(self, task):
        self.assertTrue(task.is_run_called(), 'Task run not called.')

    def test_config(self):
        task_name = self.add_simple_config()
        task = Task(task_name)
        self.assertConfigCorrectlyLoaded(task)
        self.assertIsInstance(task, Graph)

    def test_run(self):
        task = self.make_test_task()
        task.run()
        self.assertRunCalled(task)

