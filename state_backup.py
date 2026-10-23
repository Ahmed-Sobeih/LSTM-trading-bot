"""Private, hashed research-state archives with safe, non-overwriting restore."""
import argparse
import fcntl
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

import pandas as pd


KINDS = ('snapshots','decisions','outcomes','runs')
MAX_STATE_BYTES=2*1024**3
MAX_FILES=100000
MAX_MANIFEST_BYTES=16*1024**2


def clock():
    return pd.Timestamp.now(tz='UTC')


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def backup(root='data', destination='backups'):
    root, destination=Path(root),Path(destination)
    if not root.is_dir():
        raise ValueError('Research state directory does not exist')
    destination.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (root/'.daily.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files={}
        for kind in KINDS:
            directory=root/kind
            if directory.is_symlink():
                raise ValueError('Do not archive linked research directories')
            for path in sorted(directory.rglob('*')):
                if path.is_symlink():
                    raise ValueError('Do not archive symbolic links')
                if path.is_file():
                    if path.suffix not in ('.csv','.json'):
                        raise ValueError('Unexpected state file type; credentials/configuration must not enter backups')
                    files[path.relative_to(root).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
        if not files:
            raise ValueError('No research state to archive')
        created=clock()
        stamp=created.strftime('%Y%m%dT%H%M%S%fZ')
        archive=destination/f'research-{stamp}.tar.gz'
        manifest={'schema_version':1,'created_at':created.isoformat(),'files':files,
                  'warning':'Contains private market/research data, not credentials. Store privately with independent retention.'}
        payload=(json.dumps(manifest,indent=2)+'\n').encode()
        if len(files)+1>MAX_FILES or len(payload)>MAX_MANIFEST_BYTES or sum(f['bytes'] for f in files.values())+len(payload)>MAX_STATE_BYTES:
            raise ValueError('State exceeds default restore limits; review capacity before creating a new recovery point')
        with tempfile.NamedTemporaryFile(dir=destination,delete=False) as handle:
            temporary=Path(handle.name)
        try:
            with tarfile.open(temporary,'w:gz') as tar:
                info=tarfile.TarInfo('backup_manifest.json');info.size=len(payload);info.mode=0o600
                tar.addfile(info,io.BytesIO(payload))
                for name in files:
                    tar.add(root/name,arcname=name,recursive=False)
            os.chmod(temporary,0o600)
            os.link(temporary,archive)
        finally:
            temporary.unlink(missing_ok=True)
        archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'\n')
        os.chmod(archive.with_suffix(archive.suffix+'.sha256'),0o600)
        return archive


def restore(archive,destination,max_bytes=MAX_STATE_BYTES):
    archive,destination=Path(archive),Path(destination)
    checksum=archive.with_suffix(archive.suffix+'.sha256')
    if not checksum.exists() or checksum.read_text().strip()!=sha(archive):
        raise ValueError('Archive checksum missing or mismatched')
    if destination.is_symlink() or (destination.exists() and (not destination.is_dir() or any(destination.iterdir()))):
        raise ValueError('Restore destination must be absent or an empty directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        names=[member.name for member in members]
        if len(names)>MAX_FILES or len(names)!=len(set(names)) or any(m.size<0 for m in members) or sum(m.size for m in members)>max_bytes:
            raise ValueError('Duplicate/oversized archive entries')
        for member in members:
            name=PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\' in member.name:
                raise ValueError('Unsafe archive entry')
        manifest_member=tar.getmember('backup_manifest.json')
        if manifest_member.size>MAX_MANIFEST_BYTES:
            raise ValueError('Oversized archive manifest')
        manifest=json.load(tar.extractfile(manifest_member))
        if manifest.get('schema_version')!=1 or set(names)!=set(manifest['files'])|{'backup_manifest.json'}:
            raise ValueError('Archive does not match manifest')
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            staging=Path(temporary)/'state';staging.mkdir(mode=0o700)
            for name,expected in manifest['files'].items():
                if PurePosixPath(name).parts[0] not in KINDS:
                    raise ValueError('Unexpected archive state directory')
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                member=tar.getmember(name)
                if member.size!=expected['bytes']:
                    raise ValueError('Archive size mismatch')
                result=hashlib.sha256()
                with tar.extractfile(member) as source,target.open('xb') as output:
                    for block in iter(lambda:source.read(1024*1024),b''):
                        result.update(block);output.write(block)
                os.chmod(target,0o600)
                if result.hexdigest()!=expected['sha256']:
                    raise ValueError('Restored file checksum mismatch')
            os.rename(staging,destination)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    save=sub.add_parser('create');save.add_argument('--root',default='data');save.add_argument('--out',default='backups')
    load=sub.add_parser('restore');load.add_argument('archive');load.add_argument('--dest',required=True)
    args=parser.parse_args()
    print(backup(args.root,args.out) if args.command=='create' else restore(args.archive,args.dest))


if __name__=='__main__':
    main()
