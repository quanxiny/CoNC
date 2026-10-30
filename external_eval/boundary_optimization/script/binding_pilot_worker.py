"""One limited identity-only pilot unit; C input is never executed."""
from pathlib import Path
import json,sys,resource,hashlib

def main():
    q=json.loads(Path(sys.argv[1]).read_text());limit=q['limits']
    resource.setrlimit(resource.RLIMIT_CPU,(limit['cpu_seconds'],limit['cpu_seconds']+1));resource.setrlimit(resource.RLIMIT_AS,(limit['address_space_bytes'],limit['address_space_bytes']));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(32*1024**2,32*1024**2))
    raw=Path(q['source_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==q['source_sha256']
    from binding_identity import export_identity
    r=export_identity(raw.decode('utf-8'),q['unit_id'],Path(sys.argv[2]),q['entry'])
    assert r['classification_gate']=='UNKNOWN' and r['full_C_program_safety']=='UNKNOWN' and not r['facts_used_for_classification'] and not r['dataset_program_execution']
    Path(sys.argv[3]).write_text(json.dumps(r,sort_keys=True)+'\n')
if __name__=='__main__':main()
