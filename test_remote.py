import io
import json
import hashlib
import unittest
import contextlib
from pathlib import Path
from unittest.mock import patch

from botocore.exceptions import ClientError

import test_operations as fixtures
from state_backup import backup
from state_remote import pull,push,require_private_bucket
import state_remote


class FakeS3:
    def __init__(self):
        self.objects={};self.metadata={};self.public=False
    def get_public_access_block(self,**kwargs):
        return {'PublicAccessBlockConfiguration':{k:not self.public for k in
            ('BlockPublicAcls','IgnorePublicAcls','BlockPublicPolicy','RestrictPublicBuckets')}}
    def get_bucket_versioning(self,**kwargs):return {'Status':'Enabled'}
    def get_bucket_encryption(self,**kwargs):
        return {'ServerSideEncryptionConfiguration':{'Rules':[{'ApplyServerSideEncryptionByDefault':{'SSEAlgorithm':'AES256'}}]}}
    def get_object(self,Bucket,Key):
        if Key not in self.objects:raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
        data=self.objects[Key]
        return {'Body':io.BytesIO(data),'ETag':hashlib.sha256(data).hexdigest()}
    def put_object(self,Bucket,Key,Body,**kwargs):
        old=self.objects.get(Key)
        if (kwargs.get('IfNoneMatch')=='*' and old is not None) or (
            'IfMatch' in kwargs and (old is None or hashlib.sha256(old).hexdigest()!=kwargs['IfMatch'])):
            raise ClientError({'Error':{'Code':'PreconditionFailed'}},'PutObject')
        self.objects[Key]=Body.read() if hasattr(Body,'read') else Body
        self.metadata[Key]=kwargs.get('Metadata',{})
        return {}
    def head_object(self,Bucket,Key):return {'Metadata':self.metadata[Key]}
    def download_file(self,bucket,key,path):Path(path).write_bytes(self.objects[key])


class RemoteTests(unittest.TestCase):
    def setUp(self):
        fixtures.OperationsTests.setUp(self)
        self.client=FakeS3()
        self.archive=fixtures.OperationsTests.archive(self)

    def test_private_encrypted_seed_and_restore_and_no_empty_bootstrap(self):
        with self.assertRaisesRegex(ValueError,'unseeded'):
            pull(self.client,'private-test-bucket',self.base/'restored')
        data=push(self.client,'private-test-bucket',self.archive,self.root,seed=True)
        pulled=pull(self.client,'private-test-bucket',self.base/'restored')
        self.assertEqual(data,pulled)
        self.assertEqual((self.root/'decisions'/'MSFT-2024-03-15.json').read_bytes(),
                         (self.base/'restored'/'decisions'/'MSFT-2024-03-15.json').read_bytes())

    def test_public_bucket_and_corrupt_download_are_rejected(self):
        self.client.public=True
        with self.assertRaisesRegex(ValueError,'block all public'):
            require_private_bucket(self.client,'private-test-bucket')
        self.client.public=False
        data=push(self.client,'private-test-bucket',self.archive,self.root,seed=True)
        self.client.objects[data['archive_key']]+=b'corruption'
        with self.assertRaisesRegex(ValueError,'checksum'):
            pull(self.client,'private-test-bucket',self.base/'restored')
        self.assertFalse((self.base/'restored').exists())

    def test_client_initialization_failure_does_not_dump_request_details(self):
        stream=io.StringIO()
        with patch('sys.argv',['state_remote.py','pull','--bucket','private-test-bucket']), \
             patch('boto3.client',side_effect=RuntimeError('SYNTHETIC_PRIVATE_DETAIL')),contextlib.redirect_stderr(stream):
            with self.assertRaises(SystemExit) as result:
                state_remote.main()
        self.assertEqual(result.exception.code,1)
        self.assertNotIn('SYNTHETIC_PRIVATE_DETAIL',stream.getvalue())
        self.assertIn('RuntimeError',stream.getvalue())

    def test_archive_only_failure_preserves_last_good_pointer_and_cas_blocks_race(self):
        push(self.client,'private-test-bucket',self.archive,self.root,seed=True)
        restored=self.base/'restored';pull(self.client,'private-test-bucket',restored)
        old=self.client.objects['research/latest.json']
        push(self.client,'private-test-bucket',self.archive,restored,archive_only=True)
        self.assertEqual(self.client.objects['research/latest.json'],old)
        self.client.objects['research/latest.json']=old+b'\n'
        with self.assertRaises(ClientError):
            push(self.client,'private-test-bucket',self.archive,restored)
        self.assertEqual(self.client.objects['research/latest.json'],old+b'\n')


if __name__=='__main__':
    unittest.main()
