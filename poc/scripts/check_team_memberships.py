"""Read-only preflight before upgrading an existing Control Plane database.

Reports legacy projection disagreements; never migrates or clears membership.
Do not load the new reader against an inconsistent legacy database until the
operator has reviewed and resolved the report from a backed-up copy.
"""
import json
from pathlib import Path
import sqlite3
import sys


def inspect(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        people = {r['user_id']: dict(r) for r in connection.execute('SELECT * FROM control_plane_principals')}
        teams = {r['team_id']: dict(r) for r in connection.execute('SELECT * FROM control_plane_teams')}
        issues = []
        projected = {user: [] for user in people}
        for team_id, team in teams.items():
            members = json.loads(team['member_ids_json'])
            if len(set(members)) != len(members): issues.append({'team_id':team_id, 'issue':'duplicate_members'})
            for user in members:
                if user not in people or people[user]['organization_id'] != team['organization_id']:
                    issues.append({'team_id':team_id,'user_id':user,'issue':'invalid_member'})
                else: projected[user].append(team_id)
        for user, person in people.items():
            if sorted(json.loads(person['team_ids_json'])) != sorted(projected[user]):
                issues.append({'user_id':user,'issue':'legacy_projection_mismatch'})
        return {'consistent':not issues,'teams':len(teams),'members':len(people),'issues':issues}


if __name__ == '__main__':
    report = inspect(sys.argv[1])
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(0 if report['consistent'] else 1)
