"""Run offline website acceptance without opening the production Runtime DB."""
import os
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    os.environ['ROBOT_WEB_MOCK'] = '1'
    os.environ['ROBOT_WEB_NO_BROWSER'] = '1'
    with tempfile.TemporaryDirectory(prefix='website-isolation-') as directory:
        from agent.infrastructure import robot_controller
        original_output = robot_controller.WEB_OUTPUT_DIR
        robot_controller.WEB_OUTPUT_DIR = Path(directory)
        import web_app
        robot_controller.WEB_OUTPUT_DIR = original_output
        try:
            suite = unittest.defaultTestLoader.discover('.', pattern='test_*.py') if '--full' in sys.argv else unittest.TestSuite([
                unittest.defaultTestLoader.loadTestsFromName('test_website_access_policy'),
                unittest.defaultTestLoader.loadTestsFromName('test_device_assignments'),
                unittest.defaultTestLoader.loadTestsFromName('test_organization_members'),
                unittest.defaultTestLoader.loadTestsFromName('test_organization_teams'),
                unittest.defaultTestLoader.loadTestsFromName('test_organization_projects'),
                unittest.defaultTestLoader.loadTestsFromName('test_github_identity_management'),
                unittest.defaultTestLoader.loadTestsFromName('test_website_route_isolation.WebsiteRouteIsolationTests'),
                unittest.defaultTestLoader.loadTestsFromName('test_local_agent_api_client'),
            ])
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            return 0 if result.wasSuccessful() else 1
        finally:
            web_app.runtime.agent_session_repository.close()
            web_app.runtime.control_plane_repository.close()


if __name__ == '__main__':
    raise SystemExit(main())
