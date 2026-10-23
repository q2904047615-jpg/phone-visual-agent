import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image, ImageDraw
from features.lucky_bag.flow import Flow, Page
from features.lucky_bag.vision import Token, interpret, TemplateDetector
from features.lucky_bag.upstream import UpstreamDetector

class FlowTests(unittest.TestCase):
    def test_complete_loss_round_and_next_bag(self):
        flow = Flow()
        pages = [Page(bag=(40,100)),Page(detail=True,countdown=600,comment_button=(200,700)),
            Page(send_button=(400,500),prefilled=True),Page(joined=True,countdown=590)]
        phases = [flow.decide(page,10+i).phase for i,page in enumerate(pages)]
        self.assertEqual(['open_bag','open_comment','verify_join','joined'],phases)
        self.assertEqual(300,flow.decide(Page(),15).wait)
        self.assertEqual('dismiss_loss',flow.decide(Page(lost=True,dismiss_button=(200,700)),603).phase)
        result = flow.decide(Page(bag=(40,100)),606)
        self.assertEqual('open_bag',result.phase)
        self.assertFalse(result.notify)
        self.assertFalse(flow.send_attempted)

    def test_empty_prefill_stops(self):
        flow = Flow(clicked_comment=True)
        self.assertTrue(flow.decide(Page(send_button=(20,30)),0).pause)
        self.assertFalse(flow.send_attempted)

    def test_uncertain_send_never_replayed(self):
        flow = Flow(clicked_comment=True)
        self.assertIsNotNone(flow.decide(Page(send_button=(20,30),prefilled=True),0).point)
        for i in range(3):
            self.assertIsNone(flow.decide(Page(send_button=(20,30),prefilled=True),i+1).point)

    def test_deadline_without_loss_latches_stop(self):
        flow = Flow(joined=True,draw_at=20)
        step = flow.decide(Page(bag=(20,30)),20)
        self.assertTrue(step.notify)
        self.assertIsNone(step.point)
        self.assertTrue(flow.decide(Page(lost=True,dismiss_button=(10,20)),21).pause)

    def test_one_minute_when_no_bag(self):
        self.assertEqual(60,Flow().decide(Page(),0).wait)

    def test_wait_shortens_to_actual_deadline(self):
        flow = Flow(joined=True,draw_at=120)
        self.assertEqual(20,flow.decide(Page(),100).wait)

    def test_closed_room_has_no_click(self):
        step = Flow().decide(Page(closed=True,bag=(10,20)),0)
        self.assertEqual('waiting_room',step.phase)
        self.assertIsNone(step.point)

    def test_participant_count_does_not_mean_joined(self):
        self.assertFalse(interpret([Token('9192人已参与',(10,200,500,240))],(600,1200)).joined)

    def test_already_joined_and_own_countdown(self):
        page = interpret([Token('已参与',(100,800,400,850)),Token('倒计时 04:15',(100,700,400,750))],(600,1200))
        self.assertTrue(page.joined)
        self.assertEqual(255,page.countdown)

    def test_clock_and_video_timer_do_not_set_draw_deadline(self):
        page = interpret([Token('19:45',(20,10,110,40)),Token('22:37',(40,500,150,550))],(600,1200))
        self.assertIsNone(page.countdown)

    def test_placeholder_not_a_prefilled_comment(self):
        page = interpret([Token('发送',(500,700,570,740)),Token('说点什么。。。',(120,700,400,740))],(600,1200))
        self.assertFalse(page.prefilled)

    def test_prefilled_comment_can_change(self):
        page = interpret([Token('发送',(500,700,570,740)),Token('参加新的一轮！',(120,700,400,740))],(600,1200))
        self.assertTrue(page.prefilled)

    def test_loss_button_identity(self):
        page = interpret([Token('没抽中福袋',(100,200,400,250)),Token('知道了',(100,800,400,850))],(600,1200))
        self.assertTrue(page.lost)
        self.assertEqual((250,825),page.dismiss_button)

    def test_template_matches_translated_and_scaled_icon(self):
        with TemporaryDirectory() as tmp:
            icon = Image.new('RGB',(36,40),'black')
            draw = ImageDraw.Draw(icon)
            draw.rectangle((4,10,32,36),fill='red')
            draw.line((7,13,29,32),fill='yellow',width=4)
            icon.save(Path(tmp)/'icon.png')
            for scale,position in [(1,(80,120)),(1.6,(350,210))]:
                image = Image.new('RGB',(600,900),'gray')
                image.paste(icon.resize((round(36*scale),round(40*scale))),position)
                point = TemplateDetector(Path(tmp)).locate(image)
                self.assertIsNotNone(point)
                self.assertLess(abs(point[0]-position[0]-round(36*scale)/2),4)
            self.assertIsNone(TemplateDetector(Path(tmp)).locate(Image.new('RGB',(600,900),'gray')))

    def test_upstream_detector_positive_and_absent(self):
        root = Path(__file__).resolve().parents[1]
        if not (root/'external/douyin_guaji/verified-source.json').is_file():
            self.skipTest('B版不下载上游源文件')
        detector = UpstreamDetector(root/'external/douyin_guaji')
        image = Image.new('RGB',(1080,2400),'black')
        self.assertIsNone(detector.locate(image))
        image.putpixel((40,403),(190,180,240))
        self.assertIsNotNone(detector.locate(image))

if __name__ == '__main__':
    unittest.main()
