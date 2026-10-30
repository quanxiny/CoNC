"""Finite legacy-input profile and source-literal guards, without model changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PREPROCESS_TIMEOUT_SECONDS = 1.0  # Fixed before external labels/scores are read.
PROFILE = HERE / 'finite_frontend_profile.json'
# C preprocessing-number scan. Quoted text/comments are skipped, not rewritten.
# This guard never supplies model tokens; original tokenization remains intact.
LEXEME = re.compile(r'''(?P<skip>//[^\n]*|/\*[\s\S]*?\*/|"(?:\\[\s\S]|[^"\\])*"|'(?:\\[\s\S]|[^'\\])*')|(?P<number>(?<![A-Za-z0-9_])(?:[0-9]|\.[0-9])(?:[eEpP][+-]|[A-Za-z0-9_.'])*)''')
CANONICAL_DECIMAL = re.compile(r'(?:0|[1-9][0-9]*)\Z')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def literal_issues(source, included_lines=None):
    """Reject noncanonical numeric spellings only where source enters a slice.

Minus/plus stay separate original operator tokens. The guard neither evaluates
expressions nor turns unknown arithmetic or sizeof into a known value.
"""
    selected = set(included_lines) if included_lines is not None else None
    issues = []
    for match in LEXEME.finditer(source):
        if match.lastgroup != 'number':
            continue
        token = match.group()
        line = source.count('\n', 0, match.start()) + 1
        if selected is not None and line not in selected:
            continue
        if not CANONICAL_DECIMAL.fullmatch(token):
            reason = 'unsupported_literal_spelling'
        elif len(token) > 3:
            reason = 'number_outside_0_999'
        else:
            continue
        issues.append({'line': line, 'reason': reason, 'literal': token})
    return issues


def literal_fixtures():
    samples = [
        ('decimal', 'int x=0, y=9, z=999;', []),
        ('operators', 'int x=-1; int y=2+3;', []),
        ('octal', 'int x=010;', ['unsupported_literal_spelling']),
        ('leading_zero', 'int x=00;', ['unsupported_literal_spelling']),
        ('hex', 'int x=0x10;', ['unsupported_literal_spelling']),
        ('binary', 'int x=0b10;', ['unsupported_literal_spelling']),
        ('suffix', 'int x=8U; long y=9LL;', ['unsupported_literal_spelling'] * 2),
        ('float', 'double x=1.5, y=.5, z=1.;', ['unsupported_literal_spelling'] * 3),
        ('exponent', 'double x=1e-3, y=0x1p+2;', ['unsupported_literal_spelling'] * 2),
        ('range', 'int x=1000;', ['number_outside_0_999']),
        ('quoted_and_comment', 'char *s="010 1.5"; char c=\'9\'; /* 0x10 */ int v=1; // 0b11\n', []),
        ('identifier', 'int var123=3;', []),
    ]
    records = []
    for name, source, expected in samples:
        got = literal_issues(source)
        records.append({'name': name, 'expected_reasons': expected, 'observed': got,
                        'status': 'PASS' if [x['reason'] for x in got] == expected else 'FAIL'})
    return records


def verify_signed_json(path, field='result_sha256'):
    value = json.loads(Path(path).read_text())
    expected = value.pop(field, None)
    if digest(value) != expected:
        raise ValueError('Receipt self-hash mismatch: ' + str(path))
    return value


def verify_profile(path=PROFILE):
    value = verify_signed_json(path, 'profile_sha256')
    if value['status'] != 'PASS_FINITE_INPUT_PROFILE' or value['legacy_fixture_overall_status'] != 'FAIL':
        raise ValueError('Unexpected finite profile gate')
    if value['request']['driver_sha256'] != sha(__file__):
        raise ValueError('Literal/profile driver changed')
    if value['request']['legacy_driver_sha256'] != sha(HERE / 'legacy_frontend.py'):
        raise ValueError('Reviewed legacy driver changed')
    for rel, expected in value['request']['receipt_sha256'].items():
        if sha(HERE / rel) != expected:
            raise ValueError('Profile evidence receipt changed: ' + rel)
    if value['external_preprocessing_timeout_seconds'] != PREPROCESS_TIMEOUT_SECONDS:
        raise ValueError('Preprocessing deadline changed')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-profile', action='store_true', required=True)
    args = parser.parse_args()
    evidence_path = HERE / 'legacy_frontend_fixture_bounded/result.json'
    legacy = verify_signed_json(evidence_path)
    golden = legacy['completed']['author_train_golden']
    if golden['status'] != 'PASS' or not all(golden['checks'].values()):
        raise ValueError('Original author TRAIN golden parity did not pass')
    failed = {name for name, record in legacy['completed'].items() if record['status'] != 'PASS'}
    if legacy['status'] != 'FAIL' or failed != {'guard_spaces', 'cross_file_unknown'} or len(legacy['completed']) != 14:
        raise ValueError('Audit differs from reviewed finite profile; refusing to broaden scope')
    if legacy['completed']['guard_spaces'].get('error_type') != 'OriginalPreprocessingTimeout':
        raise ValueError('Spaces rejection is not the reviewed timeout')
    for record in legacy['completed'].values():
        artifact_root = Path(record.get('artifact_root', evidence_path.parent)).resolve()
        if not artifact_root.is_relative_to(HERE):
            raise ValueError('Audit artifact root escapes external_eval')
        for rel, expected in record.get('artifact_sha256', {}).items():
            if sha(artifact_root / rel) != expected:
                raise ValueError('Audit artifact changed: ' + rel)
    fixtures = literal_fixtures()
    if not all(record['status'] == 'PASS' for record in fixtures):
        raise ValueError('Literal guard self-check failed: ' + json.dumps(fixtures))
    request = {'driver_sha256': sha(__file__), 'receipt_sha256': {
        str(evidence_path.relative_to(HERE)): sha(evidence_path),
        'vendor/source_assets.json': sha(HERE / 'vendor/source_assets.json'),
        'dataset_configs.json': sha(HERE / 'dataset_configs.json')},
        'legacy_driver_sha256': legacy['request']['script_sha256']}
    value = {'status': 'PASS_FINITE_INPUT_PROFILE', 'legacy_fixture_overall_status': 'FAIL',
             'request': request, 'request_sha256': digest(request),
             'tool_and_author_train_golden_parity': 'PASS', 'golden_checks': golden['checks'],
             'compatible_fixture_checks': sorted(name for name, record in legacy['completed'].items()
                                                if record['status'] == 'PASS' and name != 'author_train_golden'),
             'retained_failed_fixtures': sorted(failed),
             'known_rejections': {
                 'guard_spaces': 'unchanged original condition helper did not terminate; preserve spaces and reject on deadline',
                 'cross_file_unknown': 'original single-file PDG produced no encodable query context; no file joining or completion',
                 'eof_no_newline': 'selected source line would lose its final character under original symbolizer',
                 'numeric_domain_rejection': 'numeric literal exceeds original 0..999 domain'},
             'unknown_policy': 'Original unknown bits, parents and sizeof/unit limitations are legal inputs when source context and input contract are intact; never replace unknown with safe.',
             'external_preprocessing_timeout_seconds': PREPROCESS_TIMEOUT_SECONDS,
             'audit_preprocessing_timeout_seconds': 3,
             'literal_policy': 'Only canonical decimal 0..999; reject multichar leading zero, octal, hex, binary, suffixes, floats, exponent forms and out-of-range values in included source lines.',
             'literal_guard_fixtures': fixtures,
             'scope': 'single-file complete-context scenes accepted by unchanged original finite preprocessing',
             'static_manifest_gate_enabled': True, 'automatic_model_inference_authorization': False,
             'source_program_proof': False, 'full_C_frontend_support': False}
    if PROFILE.exists():
        previous = verify_signed_json(PROFILE, 'profile_sha256')
        if previous == value:
            print(json.dumps({'status': 'SKIP_VERIFIED_FINITE_PROFILE', 'profile': str(PROFILE)}))
            return
        raise ValueError('Existing profile differs; refusing silent overwrite')
    value['profile_sha256'] = digest(value)
    temporary = PROFILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(PROFILE)
    print(json.dumps({'status': value['status'], 'legacy_fixture_overall_status': 'FAIL', 'profile': str(PROFILE)}))


if __name__ == '__main__':
    main()
