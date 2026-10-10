import json
import sys
from pathlib import Path
import tempfile
import unittest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oracle_higs import geometric_tile_count, run_oracle, parse_value
from capture_higs import choose_ids

class OracleTests(unittest.TestCase):
    def test_ellipse_within_single_tile(self):
        count, err = geometric_tile_count(8,8,4,4,1/16,0,1/16,1.0,2,2,16,1/255,1.0)
        self.assertIsNone(err)
        self.assertEqual(count,1)
    def test_ellipse_on_four_tile_junction(self):
        count, err = geometric_tile_count(16,16,4,4,1/16,0,1/16,1.0,2,2,16,1/255,1.0)
        self.assertIsNone(err)
        self.assertEqual(count,4)
    def test_tile_mask(self):
        import numpy as np
        m=np.array([[0,1],[1,0]], dtype=bool)
        count, err=geometric_tile_count(16,16,4,4,1/16,0,1/16,1.0,2,2,16,1/255,1.0,m)
        self.assertEqual(count,2)
    def test_aabb(self):
        count, err=geometric_tile_count(16,16,4,4,None,None,None,None,2,2,16,1/255,3.33)
        self.assertEqual(count,4)
    def test_constants(self):
        self.assertAlmostEqual(parse_value('static constexpr float GAUSSIAN_EXTEND = 3.33f;', 'GAUSSIAN_EXTEND'),3.33)
        self.assertAlmostEqual(parse_value('constexpr float ALPHA_THRESHOLD = 1.0f / 255;', 'ALPHA_THRESHOLD'),1/255)
        self.assertAlmostEqual(parse_value('static constexpr float ALPHA_THRESHOLD = 1.f / 255.f;', 'ALPHA_THRESHOLD'),1/255)
    def test_choose_ids(self):
        x=choose_ids(100,12,7)
        self.assertEqual(len(x),12)
        self.assertEqual(len(set(x)),12)
        self.assertTrue((x==choose_ids(100,12,7)).all())
    def test_captured_fixture_count_comparison(self):
        # Fixture independent from CUDA: check comparison/report/writing behavior only.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'capture.pt'
            cap={
                'means2d':torch.tensor([[[[8.,8.],[16.,16.]]]]),
                'radii':torch.tensor([[[[4,4],[4,4]]]],dtype=torch.int32),
                'depths':torch.tensor([[[1.,2.]]]),
                'conics':torch.tensor([[[[1/16,0.,1/16],[1/16,0.,1/16]]]]),
                'opacities':torch.tensor([[[1.,1.]]]),'tile_mask':None,
                'gpu_counts':torch.tensor([[[1,4]]],dtype=torch.int32),
                'sample_indices':torch.tensor([5,17]),
                'meta':{'width_in_tiles':2,'height_in_tiles':2,'tile_size':16,'n_images':1},
            }
            torch.save(cap,p)
            summary=run_oracle(p,Path(d)/'report',1/255,1)
            self.assertEqual(summary['mismatch_pairs'],0)
            self.assertEqual(summary['valid_pairs'],2)
            self.assertTrue((Path(d)/'report'/'count_comparison.csv').exists())

if __name__=='__main__':unittest.main()
