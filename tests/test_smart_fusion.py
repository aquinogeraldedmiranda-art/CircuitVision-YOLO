import unittest
from backend.smart_fusion import auto_fuse, bbox_iou, canonical

class FusionTests(unittest.TestCase):
    def test_core_predictions_are_not_replaced(self):
        core=[{'label':'Resistor','bbox':[.1,.1,.4,.4],'confidence':.71}]
        ext=[{'label':'Resistor','bbox':[.1,.1,.4,.4],'confidence':.99},
             {'label':'Oscilloscope','bbox':[.1,.1,.4,.4],'confidence':.93}]
        out=auto_fuse(core,ext,['Resistor'])
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['confidence'],.71)
        self.assertEqual(out[0]['source_model'],'core_yolo11')
    def test_new_instrument_is_added_only_at_separate_box(self):
        core=[{'label':'Resistor','bbox':[0,.1,.2,.3],'confidence':.8}]
        ext=[{'label':'Oscilloscope','bbox':[.4,.4,.9,.9],'confidence':.67}]
        out=auto_fuse(core,ext,['Resistor'])
        self.assertEqual({x['label'] for x in out},{'Resistor','Oscilloscope'})
        self.assertEqual(next(x for x in out if x['label']=='Oscilloscope')['review_status'],'experimental_extra_class')
    def test_low_confidence_and_duplicate_alias(self):
        out=auto_fuse([],[
            {'label':'7-segment','bbox':[0,0,.2,.2],'confidence':.95},
            {'label':'Function Generator','bbox':[.3,.3,.8,.8],'confidence':.4}],
            ['7 Seg','Resistor'])
        self.assertEqual(out,[])
    def test_overlapping_same_new_category_deduplicated(self):
        out=auto_fuse([],[
            {'label':'Multimeter','bbox':[.1,.1,.5,.5],'confidence':.79},
            {'label':'Multimeter','bbox':[.11,.1,.49,.49],'confidence':.74}],
            ['Resistor'])
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['confidence'],.79)
    def test_iou(self):
        self.assertEqual(bbox_iou([0,0,1,1],[0,0,1,1]),1)
        self.assertEqual(bbox_iou([0,0,.2,.2],[.8,.8,1,1]),0)

if __name__=='__main__': unittest.main()
