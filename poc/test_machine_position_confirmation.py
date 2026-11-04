"""Offline station selection contracts: no real Win32 input is dispatched."""
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from agent.infrastructure import seller_window_adapter as seller
from agent.infrastructure.robot_controller import RobotController
from test_seller_dpi_scaling import _FakeCursorUser32


def toolbar(selected, width=540, height=50):
    """Mirror the real strip: tabs over the top ~39%, a pale border row, then the control row.

    Measured on the live 810x1515 controller (strip 810x75): the selected tab is
    red on rows 0-29, row 30 is pale pink (235, 209, 209), the rest is grey.
    """
    image = Image.new('RGB', (width, height), (224, 224, 224))
    draw = ImageDraw.Draw(image)
    tab_bottom = round(height * .39)
    draw.rectangle((0, 0, width - 1, tab_bottom), fill=(185, 185, 185))
    for position in selected:
        draw.rectangle(((position - 1) * width / 10, 0,
                        position * width / 10 - 1, tab_bottom), fill=(233, 65, 62))
        draw.rectangle(((position - 1) * width / 10, tab_bottom + 1,
                        position * width / 10 - 1, tab_bottom + 2), fill=(235, 209, 209))
    return image


class StationInput(_FakeCursorUser32):
    def __init__(self):
        super().__init__((400, 400))
        self.activated = False
        self.accept_input = True

    def SetForegroundWindow(self, _hwnd):
        self.activated = True
        return 1

    def ClientToScreen(self, _hwnd, point):
        point._obj.x += 100 if self.activated else 10
        point._obj.y += 200 if self.activated else 20
        return 1

    def SendInput(self, count, events, _size):
        if not self.accept_input:
            return 0
        self.mouse_events.append(events[0].mi.dwFlags)
        return count


class MachinePositionConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.fake = StationInput()
        for context in (
            patch.object(seller, 'IS_WINDOWS', True),
            patch.object(seller, 'user32', self.fake),
            patch.object(seller, 'client_geometry', return_value=(0, 0, 540, 1010)),
            patch.object(seller, '_root_window_at', return_value=123),
            patch.object(seller.time, 'sleep'),
        ):
            context.start()
            self.addCleanup(context.stop)

    def test_control_coordinates_are_resolved_after_activation(self):
        seller.click_client_control(123, 81, 972)
        self.assertEqual((181, 1172), self.fake.positions[0])
        self.assertEqual([seller.MOUSEEVENTF_LEFTDOWN, seller.MOUSEEVENTF_LEFTUP], self.fake.mouse_events)
        self.assertEqual((400, 400), self.fake.cursor)

    def test_failed_cursor_move_never_dispatches_a_click(self):
        with patch.object(self.fake, 'SetCursorPos', return_value=0):
            with self.assertRaises(OSError):
                seller.click_client_control(123, 81, 972)
        self.assertEqual([], self.fake.mouse_events)

    def test_cursor_clipped_to_other_location_never_dispatches(self):
        with patch.object(self.fake, 'SetCursorPos', return_value=1):
            with self.assertRaisesRegex(RuntimeError, '鼠标'):
                seller.click_client_control(123, 81, 972)
        self.assertEqual([], self.fake.mouse_events)

    def test_occluded_target_never_receives_global_click(self):
        with patch.object(seller, '_root_window_at', return_value=456):
            with self.assertRaisesRegex(RuntimeError, '遮挡'):
                seller.click_client_control(123, 81, 972)
        self.assertEqual([], self.fake.mouse_events)

    def test_rejected_input_is_not_success(self):
        self.fake.accept_input = False
        with self.assertRaises(OSError):
            seller.click_client_control(123, 81, 972)

    def test_mouse_up_and_cursor_restoration_on_interrupted_hold(self):
        with patch.object(seller.time, 'sleep', side_effect=[None, RuntimeError('interrupted')]):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                seller.click_client_control(123, 81, 972)
        self.assertEqual([seller.MOUSEEVENTF_LEFTDOWN, seller.MOUSEEVENTF_LEFTUP], self.fake.mouse_events)
        self.assertEqual((400, 400), self.fake.cursor)

    def test_toolbar_reading_all_stations_scales_and_landscape(self):
        for width, height in ((540, 50), (675, 63), (810, 75), (1440, 75)):
            for position in range(1, 11):
                with self.subTest(size=(width, height), position=position):
                    self.assertEqual(position, seller.machine_position_from_toolbar(toolbar([position], width, height)))
        for selected in ([], [1, 2]):
            self.assertIsNone(seller.machine_position_from_toolbar(toolbar(selected)))

    def test_selected_tab_is_read_inside_the_tab_not_on_its_border(self):
        # Regression (2026-11-05): button 1 was red on the real controller, but the
        # 40% sample row hit the pale border, scoring 4/6, so the switch was
        # reported as MachinePositionNotConfirmed.
        image = Image.new('RGB', (810, 75), (224, 224, 224))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 809, 29), fill=(185, 185, 185))
        for row in range(28):
            draw.line((0, row, 80, row), fill=(246 - row // 2, 90 - row, 89 - row))
        draw.line((0, 28, 80, 28), fill=(198, 40, 40))
        draw.line((0, 29, 80, 29), fill=(198, 40, 40))
        draw.line((0, 30, 80, 30), fill=(235, 209, 209))
        self.assertEqual(1, seller.machine_position_from_toolbar(image))
        for y in seller.TOOLBAR_TAB_SAMPLE_ROWS:
            self.assertLess(int(y * 75), 30)

    def test_toolbar_read_is_passive_and_excludes_phone_pixels(self):
        with (patch.object(seller, '_window_is_minimized', return_value=False),
              patch.object(seller.ImageGrab, 'grab', return_value=toolbar([2])) as grab):
            self.assertEqual(2, seller.read_machine_position(123))
        grab.assert_called_once_with(bbox=(0, 960, 540, 1010), all_screens=True)
        self.assertFalse(self.fake.activated)
        self.assertEqual([], self.fake.mouse_events)

    def test_unreadable_toolbar_is_unknown_not_a_requested_station(self):
        with (patch.object(seller, '_window_is_minimized', return_value=False),
              patch.object(seller, '_root_window_at', return_value=456),
              patch.object(seller.ImageGrab, 'grab') as grab):
            self.assertIsNone(seller.read_machine_position(123))
            grab.assert_not_called()
        with patch.object(seller, '_window_is_minimized', return_value=True):
            self.assertIsNone(seller.read_machine_position(123))

    def test_no_selection_change_is_failure_not_success(self):
        with (patch.object(seller, 'ensure_window_fully_visible'),
              patch.object(seller, 'click_client_control') as click,
              patch.object(seller, 'read_machine_position', return_value=1)):
            with self.assertRaisesRegex(seller.MachinePositionNotConfirmed, '2'):
                seller.select_machine_position(123, 2)
            click.assert_called_once()

    def test_delayed_selection_does_not_repeat_click(self):
        with (patch.object(seller, 'ensure_window_fully_visible'),
              patch.object(seller, 'click_client_control') as click,
              patch.object(seller, 'read_machine_position', side_effect=[1, None, 2])):
            seller.select_machine_position(123, 2)
            click.assert_called_once()

    def test_failed_switch_clears_ready_without_committing_requested_station(self):
        controller = RobotController(title='test', machine_position=1, device_id='test-device')
        controller._machine_position_ready = True
        with (patch.object(seller, 'find_window', return_value=(123, 'test')),
              patch.object(seller, 'select_machine_position', side_effect=RuntimeError('unconfirmed'))):
            with self.assertRaises(RuntimeError):
                controller.select_machine_position(2)
        self.assertEqual(1, controller.machine_position)
        self.assertFalse(controller._machine_position_ready)


if __name__ == '__main__':
    unittest.main()
