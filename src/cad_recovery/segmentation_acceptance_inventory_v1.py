"""Canonical file recipe for the frozen expanded acceptance producer."""
from .dataset_layout_v1 import pin
from .family_catalog_v2 import require


def expected_files(items,new_ids,variants):
    names={'inputs.json','report.json','index.html','cv-runtime.json',
           'cv-process/stdout.txt','cv-process/stderr.txt','cv/results.json'}
    for item in items:
        names.update((item['image_name'],item['target_name']))
        names.update('preview/'+item['id']+'-'+s+'.png' for s in ('source','target','prediction'))
    for variant in variants:
        names.update(variant+'/'+s for s in ('runtime.json','result/result.json','process/stdout.txt','process/stderr.txt'))
        for item in items:
            if item['kind']=='patch' or variant in ('pretrained-gray','pretrained-gray-repeat'):
                names.add(variant+'/result/'+item['id']+'.png')
    for name in new_ids:
        names.add('cv/'+name+'/guided.png')
        for variant in ('original','guided'):
            names.update('cv/'+name+'/'+variant+'/'+s for s in ('page-001-geometry.json',
                'page-001-geometry-preview.png','geometry-canonical.json','arcs.json','heads.json'))
    return names


def validate(folder,manifest,items,new_ids,variants):
    expected=expected_files(items,new_ids,variants)
    require(manifest['files']==[pin(folder,n) for n in sorted(expected)],'Changed canonical acceptance file recipe')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual==expected|{'manifest.json'},'Unexpected acceptance file outside recipe')
    return len(expected)
