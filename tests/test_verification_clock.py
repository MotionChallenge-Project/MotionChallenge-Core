import unittest
from motion_challenge.verify import failure_time_matches


class VerificationClockTests(unittest.TestCase):
    def test_integrated_physics_roundoff_matches_metric_clock(self):
        self.assertTrue(failure_time_matches(20.99999999999723,21.0,True))

    def test_shifted_missing_or_nonfinite_failure_time_is_rejected(self):
        for value in (21.000001,20.98,None,float('nan'),float('inf')):
            self.assertFalse(failure_time_matches(value,21.0,True))
        self.assertFalse(failure_time_matches(21.0,21.0,False))
        self.assertTrue(failure_time_matches(None,21.0,False))


if __name__=='__main__':unittest.main()
