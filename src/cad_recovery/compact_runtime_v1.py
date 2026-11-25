"""Explicit runtime identity, separated from historical qualification witnesses.

Only profile/table metadata factories are adapted, before consumers import them.
File I/O/hash helpers, algorithms, model checks and numerical rules are unchanged.
"""
from copy import deepcopy
import gzip,hashlib,importlib,importlib.metadata,json,os,sys,threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
_active=None;_lock=threading.RLock()
MANAGED=('src','scripts','configs','runtime','artifacts/models',
    'artifacts/service-runtime-v1/deps','artifacts/cv-skeleton-runtime-v1/deps','artifacts/service-postgres-v1/pgsql')
TRUST='src/cad_recovery/compact_trust_v1.py'


class CompactContractError(ValueError):pass


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def inside(root,name):
    if not isinstance(name,str) or not name or '\\' in name or ':' in name or any(x in ('','.','..') for x in name.split('/')):
        raise CompactContractError('Invalid compact path')
    path=root/name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):raise CompactContractError('Compact path escapes root')
    for p in (path,*path.parents):
        if p==root:break
        if p.is_symlink() or getattr(p,'is_junction',lambda:False)():raise CompactContractError('Linked compact path')
    return path


def verify_files(root,index,*,full=False):
    root=root.resolve()
    rows=index['files'];expected={r['path'].casefold():r for r in rows}
    if len(expected)!=len(rows):raise CompactContractError('Duplicate compact destination')
    for row in rows:
        name=row['path']
        if not name or '\\' in name or ':' in name or Path(name).is_absolute() or any(x in ('','.','..') for x in name.split('/')):
            raise CompactContractError('Invalid compact path')
    # Validate each directory/link exactly once. Rewalking every ancestor for
    # every file turns a 10k-file inventory into hundreds of thousands of stats.
    actual={}
    def scan(directory):
        if directory.is_symlink() or getattr(directory,'is_junction',lambda:False)():raise CompactContractError('Linked runtime directory')
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink() or getattr(entry,'is_junction',lambda:False)():raise CompactContractError('Linked runtime directory')
                if entry.is_dir(follow_symlinks=False):scan(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    path=Path(entry.path);actual[path.relative_to(root).as_posix().casefold()]=(path,entry.stat(follow_symlinks=False))
    for directory in MANAGED:
        parent=root/directory
        if parent.exists():
            for ancestor in (parent,*parent.parents):
                if ancestor==root:break
                if ancestor.is_symlink() or getattr(ancestor,'is_junction',lambda:False)():raise CompactContractError('Linked runtime directory')
            scan(parent)
    allowed=set(expected)|{TRUST.casefold(),'runtime/compact/index.json'}
    extra=set(actual)-allowed
    if extra:raise CompactContractError('Unexpected runtime file: '+sorted(extra)[0])
    for row in rows:
        if not full and row['kind']=='environment':continue
        item=actual.get(row['path'].casefold())
        if item is None:
            path=inside(root,row['path'])
            item=(path,path.stat()) if path.is_file() else None
        if item is None or item[1].st_size!=row['size_bytes'] or digest(item[0])!=row['sha256']:
            raise CompactContractError('Runtime identity differs: '+row['path'])
    return len(rows)


def load(root,index_sha):
    path=inside(root,'runtime/compact/index.json')
    if digest(path)!=index_sha:raise CompactContractError('Compact index SHA differs')
    index=json.loads(path.read_text(encoding='utf-8'))
    if index.get('schema_version')!='compact-runtime/1.0':raise CompactContractError('Unknown compact schema')
    verify_files(root,index,full=True)
    compressed=inside(root,'runtime/compact/bindings.json.gz')
    with gzip.open(compressed,'rb') as f:data=json.loads(f.read(32*1024**2+1))
    if data['schema_version']!='compact-bindings/1.0':raise CompactContractError('Unknown binding schema')
    for package,version in index['dependencies'].items():
        if importlib.metadata.version(package)!=version:raise CompactContractError('Dependency version differs: '+package)
    font=index.get('font')
    if font:
        p=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/font['file']
        if not p.is_file() or digest(p)!=font['sha256']:raise CompactContractError('CAD font differs')
    return index,data


def binding():
    if _active is None:raise CompactContractError('Compact runtime is not active')
    return dict(schema_version='compact-runtime-binding/1.0',index_sha256=_active['sha'],
        historical_witnesses_required=False,legacy_profile_identifiers_preserved=True)


def verify_current():
    if _active is None:raise CompactContractError('Compact runtime is not active')
    return verify_files(ROOT,_active['index'])


def verify_environment():
    if _active is None:raise CompactContractError('Compact runtime is not active')
    return verify_files(ROOT,_active['index'],full=True)


def resolve(version,name):
    verify_current()
    try:return deepcopy(_active['data']['profiles'][str(version)][name])
    except KeyError:
        from .profile_contract_v1 import ProfileError
        raise ProfileError('profile_unknown','Profile is unavailable in this runtime') from None


def validate(version,name,value):
    if version>=6 and value and value.get('schema_version')!=f'processing-profile/{version}.0':
        return validate(version-1,name,value)
    current=resolve(version,name)
    if value!=current:
        from .profile_contract_v1 import ProfileError
        raise ProfileError('profile_contract_changed','Stored profile differs; explicit reprocessing is required')
    return current


def table_binding(version):
    verify_current();return deepcopy(_active['data']['tables'][str(version)])


def activate():
    global _active
    with _lock:
        if _active is not None:return binding()
        from .compact_trust_v1 import INDEX_SHA256
        index,data=load(ROOT,INDEX_SHA256)
        profiles=[importlib.import_module(f'cad_recovery.profile_contract_v{i}') for i in range(1,10)]
        tables=[importlib.import_module(f'cad_recovery.web_tables_contract_v{i}') for i in range(1,8)]
        # Late activation would leave imported local aliases pointing at the
        # historical factories. Fail instead of mixing verification policies.
        if any(name.startswith('cad_recovery.profile_pipeline_') for name in sys.modules):
            raise CompactContractError('Activate compact runtime before pipeline imports')
        _active=dict(index=index,data=data,sha=INDEX_SHA256)
        for i,module in enumerate(profiles,1):
            def resolver(name,version=i):return resolve(version,name)
            def validator(name,value,version=i):return validate(version,name,value)
            module.resolve_profile=resolver;module.validate_binding=validator
            if hasattr(module,'verify_cv_runtime'):module.verify_cv_runtime=verify_environment
        for i,module in enumerate(tables,1):
            def factory(version=i):return table_binding(version)
            module.binding=factory
        return binding()
