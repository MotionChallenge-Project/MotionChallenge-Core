import copy
import unittest

from motion_challenge.merge import CONTRACT_KEYS, validate_shards


class ShardTests(unittest.TestCase):
    def records(self):
        common = {k:'same-frozen-contract' for k in CONTRACT_KEYS}
        return [{**common,'status':'completed','clips':[{'id':name,'truncated_smoke':False}]}
                for name in ('b','a')]

    def test_reordered_shards_are_restored_to_frozen_suite_order(self):
        result=validate_shards(self.records(),['a','b'])
        self.assertEqual([c['id'] for c in result],['a','b'])

    def test_different_checkpoint_cannot_form_one_full_suite(self):
        records=self.records();records[1]['controller_config_sha256']='different'
        with self.assertRaisesRegex(ValueError,'different evaluation'):validate_shards(records,['a','b'])

    def test_duplicate_missing_or_truncated_motion_cannot_rank(self):
        records=self.records()
        with self.assertRaisesRegex(ValueError,'duplicate'):validate_shards(records+records[:1],['a','b'])
        with self.assertRaisesRegex(ValueError,'missing'):validate_shards(records[:1],['a','b'])
        records[0]['clips'][0]['truncated_smoke']=True
        with self.assertRaisesRegex(ValueError,'smoke'):validate_shards(records,['a','b'])


if __name__=='__main__':unittest.main()
