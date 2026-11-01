#!/usr/bin/env python3
"""Compare a transferred local manifest against this shared project checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
rows=json.loads(a.manifest.read_text());bad=[]
for r in rows:
 f=ROOT/r['path']
 if not f.is_file():bad.append({'path':r['path'],'error':'missing'});continue
 if f.stat().st_size!=r['size'] or hashlib.file_digest(f.open('rb'),'sha256').hexdigest()!=r['sha256']:bad.append({'path':r['path'],'error':'hash mismatch'})
result={'host':os.uname().nodename,'passed':not bad,'checked_files':len(rows),'mismatches':bad}
if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
print(json.dumps({'host':result['host'],'passed':not bad,'checked_files':len(rows),'mismatches':len(bad)}))
raise SystemExit(0 if not bad else 2)
