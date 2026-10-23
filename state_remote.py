"""Private S3 state transport. Uses host credentials/OIDC; never stores keys."""
import argparse
import base64
import json
import tempfile
from pathlib import Path, PurePosixPath

from botocore.exceptions import ClientError

from state_backup import restore,sha


def require_private_bucket(client,bucket):
    block=client.get_public_access_block(Bucket=bucket)['PublicAccessBlockConfiguration']
    if not all(block.get(k) for k in ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')):
        raise ValueError('State bucket must block all public access')
    if client.get_bucket_versioning(Bucket=bucket).get('Status')!='Enabled':
        raise ValueError('State bucket versioning must be enabled')
    rules=client.get_bucket_encryption(Bucket=bucket)['ServerSideEncryptionConfiguration']['Rules']
    if not any(r.get('ApplyServerSideEncryptionByDefault',{}).get('SSEAlgorithm')=='AES256' for r in rules):
        raise ValueError('This transport requires default AES256 encryption; KMS needs a separately configured policy')


def pointer(client,bucket,prefix):
    try:
        response=client.get_object(Bucket=bucket,Key=prefix+'/latest.json')
    except ClientError as error:
        if error.response.get('Error',{}).get('Code') in ('NoSuchKey','404'):
            return None,None
        raise
    payload=response['Body'].read(65537)
    if len(payload)>65536:
        raise ValueError('Oversized remote state pointer')
    data=json.loads(payload)
    key=PurePosixPath(data['archive_key'])
    if data.get('schema_version')!=1 or not str(key).startswith(prefix+'/archives/') or '..' in key.parts or key.is_absolute():
        raise ValueError('Invalid remote state pointer')
    checksum=data.get('sha256','')
    if len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('Invalid remote archive checksum')
    return data,response['ETag']


def pull(client,bucket,root='data',prefix='research'):
    require_private_bucket(client,bucket)
    data,etag=pointer(client,bucket,prefix)
    if data is None:
        raise ValueError('Remote state is unseeded; upload the existing research archive before enabling automation')
    with tempfile.TemporaryDirectory() as temporary:
        archive=Path(temporary)/Path(data['archive_key']).name
        client.download_file(bucket,data['archive_key'],str(archive))
        if sha(archive)!=data['sha256']:
            raise ValueError('Downloaded archive checksum mismatch')
        archive.with_suffix(archive.suffix+'.sha256').write_text(data['sha256'])
        restore(archive,root)
    (Path(root)/'.remote-pointer.json').write_text(json.dumps({'etag':etag,'bucket':bucket,'prefix':prefix}))
    return data


def push(client,bucket,archive,root='data',prefix='research',archive_only=False,seed=False):
    require_private_bucket(client,bucket)
    archive=Path(archive)
    checksum=sha(archive)
    if archive.with_suffix(archive.suffix+'.sha256').read_text().strip()!=checksum:
        raise ValueError('Local archive checksum mismatch')
    key=prefix+'/archives/'+archive.name
    if not archive.name.startswith('research-') or not archive.name.endswith('.tar.gz'):
        raise ValueError('Unexpected backup archive filename')
    try:
        with archive.open('rb') as body:
            client.put_object(Bucket=bucket,Key=key,Body=body,ContentLength=archive.stat().st_size,
                              ServerSideEncryption='AES256',IfNoneMatch='*',
                              ChecksumSHA256=base64.b64encode(bytes.fromhex(checksum)).decode(),Metadata={'sha256':checksum})
    except ClientError as error:
        if error.response.get('Error',{}).get('Code') not in ('PreconditionFailed','412'):
            raise
        if client.head_object(Bucket=bucket,Key=key).get('Metadata',{}).get('sha256')!=checksum:
            raise ValueError('Remote archive already exists with different bytes')
    data={'schema_version':1,'archive_key':key,'sha256':checksum}
    if not archive_only:
        if seed:
            condition={'IfNoneMatch':'*'}
        else:
            local=json.loads((Path(root)/'.remote-pointer.json').read_text())
            if local['bucket']!=bucket or local['prefix']!=prefix:
                raise ValueError('Remote state binding changed')
            condition={'IfMatch':local['etag']}
        client.put_object(Bucket=bucket,Key=prefix+'/latest.json',Body=(json.dumps(data)+'\n').encode(),
                          ContentType='application/json',ServerSideEncryption='AES256',**condition)
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['pull','push'])
    parser.add_argument('--bucket',required=True);parser.add_argument('--root',default='data')
    parser.add_argument('--prefix',default='research');parser.add_argument('--archive')
    parser.add_argument('--archive-only',action='store_true');parser.add_argument('--seed',action='store_true')
    args=parser.parse_args()
    if not args.prefix or any(p in ('','.','..') for p in args.prefix.split('/')):
        parser.error('Use a nonempty relative S3 prefix')
    import boto3
    try:
        client=boto3.client('s3')
        if args.command=='pull':
            result=pull(client,args.bucket,args.root,args.prefix)
        else:
            archive=Path(args.archive) if args.archive else max(Path('backups').glob('research-*.tar.gz'))
            result=push(client,args.bucket,archive,args.root,args.prefix,args.archive_only,args.seed)
        print(json.dumps(result,indent=2))
    except Exception as error:
        # AWS exceptions can include request/account details: report class only.
        parser.exit(1,f'Private state transport failed ({type(error).__name__}); verify bucket policy, OIDC role and seeded state.\n')


if __name__=='__main__':
    main()
