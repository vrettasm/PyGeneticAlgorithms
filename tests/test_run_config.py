import unittest
from dataclasses import is_dataclass
from pygenalgo.engines.generic_ga import RunConfig

class TestRunConfig(unittest.TestCase):

    def test_default_values(self):
        config = RunConfig()

        self.assertEqual(config.epochs, 100)
        self.assertTrue(config.elitism)
        self.assertTrue(config.shuffle)
        self.assertFalse(config.correction)
        self.assertFalse(config.adapt_probs)
        self.assertFalse(config.parallel)
        self.assertFalse(config.verbose)
        self.assertIsNone(config.f_tol)
        self.assertIsNone(config.f_max_eval)
        self.assertFalse(config.allow_migration)
    # _end_def_

    def test_custom_values(self):
        config = RunConfig(
            epochs=50,
            elitism=False,
            shuffle=False,
            correction=True,
            adapt_probs=True,
            parallel=True,
            verbose=True,
            f_tol=0.001,
            f_max_eval=1000,
            allow_migration=True
        )

        self.assertEqual(config.epochs, 50)
        self.assertFalse(config.elitism)
        self.assertFalse(config.shuffle)
        self.assertTrue(config.correction)
        self.assertTrue(config.adapt_probs)
        self.assertTrue(config.parallel)
        self.assertTrue(config.verbose)
        self.assertEqual(config.f_tol, 0.001)
        self.assertEqual(config.f_max_eval, 1000)
        self.assertTrue(config.allow_migration)
    # _end_def_

    def test_is_frozen(self):
        config = RunConfig()
        self.assertTrue(is_dataclass(config))
        self.assertTrue(config.__class__.__dataclass_params__.frozen)
    # _end_def_

if __name__ == '__main__':
    unittest.main()
