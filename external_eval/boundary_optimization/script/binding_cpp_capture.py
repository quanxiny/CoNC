"""Transparent bounded-probe CPP recorder; forwards only to existing GCC -E.

Not a compiler replacement: records actual argv/output/header contents before
Frama deletes temporary machdep files. Never runs a compiled source program.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    args=sys.argv[1:]
    if '-E' not in args or '-o' not in args or any(x in args for x in ('-c','-S')):
        raise RuntimeError('CAPTURE_ONLY_PREPROCESSING_ALLOWED')
    folder=Path(os.environ['BINDING_CPP_CAPTURE_DIR'])/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    compiler=os.environ['BINDING_CPP_COMPILER']
    command=[compiler,*args]
    result=subprocess.run(command,capture_output=True,timeout=55)
    sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)
    (folder/'stdout').write_bytes(result.stdout);(folder/'stderr').write_bytes(result.stderr)
    output=Path(args[args.index('-o')+1]);headers=[]
    for line in result.stderr.decode(errors='replace').splitlines():
        if line.startswith('.'):
            p=Path(line.lstrip('.').strip())
            if p.is_file():headers.append({'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    if output.exists():(folder/'output.i').write_bytes(output.read_bytes())
    for arg in args:
        if arg.startswith('-I/tmp/__fc_machdep'):
            for p in Path(arg[2:]).glob('*'):
                if p.is_file():(folder/p.name).write_bytes(p.read_bytes())
    receipt={'command':command,'returncode':result.returncode,'headers':headers,
        'compiler_sha256':hashlib.sha256(Path(compiler).read_bytes()).hexdigest(),
        'source_execution':False}
    (folder/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    return result.returncode


if __name__=='__main__':sys.exit(main())
