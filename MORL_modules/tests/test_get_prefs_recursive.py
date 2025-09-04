import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_algos.hyper_morl_utils import get_prefs_recursive


class TestGetPrefsRecursive(unittest.TestCase):
    """
    Unit tests for the get_prefs_recursive function.

    The function generates preference vectors with the following properties:
    - k: granularity parameter (number of divisions)
    - m: number of dimensions/objectives
    - s: sum constraint (all preferences should sum to s)
    """

    def setUp(self):
        """Set up test fixtures with tolerance for floating point comparisons."""
        self.tolerance = 1e-10

    def test_base_case_m1(self):
        """Test base case when m=1 (single objective)."""
        result = get_prefs_recursive(k=5, m=1, s=1.0)

        # Should return single preference vector with value s
        self.assertEqual(len(result), 1)
        np.testing.assert_array_almost_equal(result[0], np.array([1.0]))

        # Test with different s value
        result_s2 = get_prefs_recursive(k=3, m=1, s=2.0)
        self.assertEqual(len(result_s2), 1)
        np.testing.assert_array_almost_equal(result_s2[0], np.array([2.0]))

    def test_m2_case(self):
        """Test 2-dimensional case (m=2)."""
        k, m, s = 2, 2, 1.0
        result = get_prefs_recursive(k, m, s)

        # Should have k+1 preference vectors
        expected_length = k + 1
        self.assertEqual(len(result), expected_length)

        # Check each preference vector
        r = s / k  # 0.5
        expected_prefs = [
            np.array([0.0, 1.0]),  # [0*r, s-0*r]
            np.array([0.5, 0.5]),  # [1*r, s-1*r]
            np.array([1.0, 0.0])   # [2*r, s-2*r]
        ]

        for i, expected_pref in enumerate(expected_prefs):
            np.testing.assert_array_almost_equal(result[i], expected_pref)

    def test_m3_case(self):
        """Test 3-dimensional case (m=3)."""
        k, m, s = 2, 3, 1.0
        result = get_prefs_recursive(k, m, s)

        # For m=3, k=2: should have (k+1)*(k+2)/2 = 6 preference vectors
        expected_length = 6
        self.assertEqual(len(result), expected_length)

        # Check that all vectors have 3 dimensions
        for pref in result:
            self.assertEqual(len(pref), 3)

    def test_sum_constraint(self):
        """Test that all preference vectors sum to s."""
        test_cases = [
            (2, 2, 1.0),
            (3, 3, 1.0),
            (2, 4, 2.0),
            (1, 5, 0.5)
        ]

        for k, m, s in test_cases:
            with self.subTest(k=k, m=m, s=s):
                result = get_prefs_recursive(k, m, s)
                for pref in result:
                    pref_sum = np.sum(pref)
                    self.assertAlmostEqual(pref_sum, s, places=10,
                                         msg=f"Preference {pref} doesn't sum to {s}")

    def test_non_negative_values(self):
        """Test that all preference values are non-negative."""
        test_cases = [
            (2, 2, 1.0),
            (3, 3, 1.0),
            (2, 4, 2.0),
            (4, 2, 1.0)
        ]

        for k, m, s in test_cases:
            with self.subTest(k=k, m=m, s=s):
                result = get_prefs_recursive(k, m, s)
                for pref in result:
                    self.assertTrue(np.all(pref >= -self.tolerance),
                                  msg=f"Negative values found in preference {pref}")

    def test_correct_dimensions(self):
        """Test that all preference vectors have correct dimensions."""
        test_cases = [
            (2, 2, 1.0),
            (3, 3, 1.0),
            (2, 4, 2.0),
            (1, 5, 0.5)
        ]

        for k, m, s in test_cases:
            with self.subTest(k=k, m=m, s=s):
                result = get_prefs_recursive(k, m, s)
                for pref in result:
                    self.assertEqual(len(pref), m,
                                   msg=f"Expected {m} dimensions, got {len(pref)}")

    def test_k_zero_edge_case(self):
        """Test edge case when k=0."""
        # When k=0, r = s/k would cause division by zero
        # The function should handle this gracefully
        with self.assertRaises(ZeroDivisionError):
            get_prefs_recursive(k=0, m=2, s=1.0)

    def test_recursive_case_m4(self):
        """Test recursive case for m>3."""
        k, m, s = 2, 4, 1.0
        result = get_prefs_recursive(k, m, s)

        # Check that result is not empty
        self.assertGreater(len(result), 0)

        # Check dimensions and sum constraint
        for pref in result:
            self.assertEqual(len(pref), 4)
            self.assertAlmostEqual(np.sum(pref), s, places=10)

    def test_specific_known_case_m2_k3(self):
        """Test a specific case with known expected output."""
        k, m, s = 3, 2, 1.0
        result = get_prefs_recursive(k, m, s)

        # Should have 4 preference vectors for k=3, m=2
        self.assertEqual(len(result), 4)

        expected_prefs = [
            np.array([0.0, 1.0]),
            np.array([1/3, 2/3]),
            np.array([2/3, 1/3]),
            np.array([1.0, 0.0])
        ]

        for i, expected_pref in enumerate(expected_prefs):
            np.testing.assert_array_almost_equal(result[i], expected_pref, decimal=10)

    def test_specific_known_case_m3_k2(self):
        """Test a specific 3D case with known expected output."""
        k, m, s = 2, 3, 1.0
        result = get_prefs_recursive(k, m, s)

        # Should have 6 preference vectors
        self.assertEqual(len(result), 6)

        # All should sum to 1 and have 3 dimensions
        for pref in result:
            self.assertEqual(len(pref), 3)
            self.assertAlmostEqual(np.sum(pref), 1.0, places=10)
            self.assertTrue(np.all(pref >= -self.tolerance))

    def test_different_s_values(self):
        """Test function behavior with different s values."""
        s_values = [0.5, 1.0, 2.0, 10.0]
        k, m = 2, 3

        for s in s_values:
            with self.subTest(s=s):
                result = get_prefs_recursive(k, m, s)

                # Check that all preferences sum to s
                for pref in result:
                    self.assertAlmostEqual(np.sum(pref), s, places=10)

    def test_return_type_is_list_of_arrays(self):
        """Test that the function returns a list of numpy arrays."""
        result = get_prefs_recursive(k=2, m=2, s=1.0)

        self.assertIsInstance(result, list)
        for pref in result:
            self.assertIsInstance(pref, np.ndarray)

    def test_empty_result_edge_cases(self):
        """Test cases that might result in empty or minimal results."""
        # Test with k=1, m=2
        result = get_prefs_recursive(k=1, m=2, s=1.0)
        self.assertEqual(len(result), 2)  # Should have k+1 = 2 preferences

        # Test with very small k and large m
        result = get_prefs_recursive(k=1, m=5, s=1.0)
        self.assertGreater(len(result), 0)  # Should still produce some preferences

    def test_numerical_stability(self):
        """Test numerical stability with larger k values."""
        k, m, s = 10, 3, 1.0
        result = get_prefs_recursive(k, m, s)

        # Check that sums are still accurate despite more floating point operations
        for pref in result:
            self.assertAlmostEqual(np.sum(pref), s, places=8)  # Slightly relaxed tolerance


if __name__ == '__main__':
    unittest.main()
