"""Partial-target diagnostics: never infer full-page truth from missing labels."""
import cv2
import numpy as np

from . import segmentation_experiment_v1 as experiment
from .family_catalog_v2 import require

POLICY={'version':'expanded-segmentation-acceptance/1.0','tile_side':256,'stride':192,
    'primary':'pretrained-gray','variants':['pretrained-gray','random-gray','pretrained-rgb','pretrained-gray-repeat'],
    'geometry_classes':[1,2,3,4,5,9],'minimum_component_pixels':24,'minimum_component_extent':16,
    'thin_maximum_radius_px':4,'component_retention_fraction':.8,'primitive_band_px':3,
    'full_view_sources':['train-extra-1','validation-new-1','validation-new-2'],
    'maximum_full_view_pixels':16777216,'max_child_seconds':900,'max_host_MiB':6144,
    'max_allocated_MiB':4096,'max_reserved_MiB':6144,'max_device_MiB':8192,
    'necessary_partial_gates':{'minimum_each_foreground_IoU':.5,'maximum_background_false_foreground':.01,
        'minimum_thin_axis_connected_80_fraction':.8},
    'guided_geometry':'keep original RGB only where primary argmax is in classes 1,2,3,4,5,9; white elsewhere',
    'arrow_source':'path proposals may be filtered; head verification always uses original native gray',
    'new_training':False,'validation_model_selection':False,'closed_test_used':False,
    'physical_geometry_approved':False,'production_admitted':False,'independent_evaluation':False}


def islands(target,prediction):
    """Connected fragments inside each known reference island, not CAD topology."""
    result=[]
    for role in POLICY['geometry_classes']:
        reference=(target==role).astype(np.uint8)
        count,labels,stats,_=cv2.connectedComponentsWithStats(reference,connectivity=8)
        kept=((target==role)&(prediction==role)).astype(np.uint8)
        n,connected,cstats,_=cv2.connectedComponentsWithStats(kept,connectivity=8)
        maxima=np.zeros(count,np.int64);pieces=np.zeros(count,np.int64)
        if n>1:
            indices=np.flatnonzero(kept.ravel())
            ids,first=np.unique(connected.ravel()[indices],return_index=True)
            owners=labels.ravel()[indices[first]]
            np.maximum.at(maxima,owners,cstats[ids,cv2.CC_STAT_AREA])
            np.add.at(pieces,owners,1)
        distance=cv2.distanceTransform(reference,cv2.DIST_L2,3)
        radii=np.zeros(count,float);np.maximum.at(radii,labels.ravel(),distance.ravel())
        retained=np.bincount(labels.ravel(),weights=kept.ravel(),minlength=count)
        for index in range(1,count):
            x,y,w,h,area=map(int,stats[index])
            if area<POLICY['minimum_component_pixels'] or max(w,h)<POLICY['minimum_component_extent']:continue
            result.append({'role':role,'box':[x,y,x+w,y+h],'reference_pixels':area,
                'retained_pixels':int(retained[index]),'fragments':int(pieces[index]),
                'largest_connected_pixels':int(maxima[index]),'largest_connected_fraction':float(maxima[index]/area),
                'thin':bool(radii[index]<=POLICY['thin_maximum_radius_px'])})
    return result


def summarize(matrix,components):
    result=experiment.metrics(matrix)
    background=int(np.asarray(matrix)[0].sum())
    result['background_false_foreground_fraction']=float(np.asarray(matrix)[0,1:].sum()/background) if background else None
    thin=[c for c in components if c['role']==5 and c['thin']]
    result['reference_island_count']=len(components)
    result['thin_axis_islands']=len(thin)
    result['thin_axis_connected_80_fraction']=float(np.mean([c['largest_connected_fraction']>=.8 for c in thin])) if thin else None
    result['topology_scope']='largest correctly classified connected fragment inside known ink islands; unknown regions and full CAD topology unmeasured'
    return result


def evaluate(target,prediction):
    matrix=experiment.confusion(target,prediction);components=islands(target,prediction)
    return {'metrics':summarize(matrix,components),'components':components}


def combine_evaluations(items):
    require(bool(items),'Empty evaluation stratum')
    matrix=np.sum([np.asarray(v['metrics']['confusion'],np.int64) for v in items],axis=0)
    return summarize(matrix,[c for v in items for c in v['components']])


def guided_rgb(rgb,prediction):
    require(rgb.dtype==np.uint8 and prediction.shape==rgb.shape[:2],'Invalid native guided input')
    return np.where(np.isin(prediction,POLICY['geometry_classes'])[:,:,None],rgb,255).astype(np.uint8)


def merged_targets(rows,arrays,source,box):
    left,top,right,bottom=box;result=np.full((bottom-top,right-left),255,np.uint8)
    conflicts=np.zeros(result.shape,bool)
    for row in rows:
        if row['source']!=source:continue
        x,y=row['patch_to_source'][0][2],row['patch_to_source'][1][2]
        w,h=row['size_px'];l,t,r,b=max(left,x),max(top,y),min(right,x+w),min(bottom,y+h)
        if l>=r or t>=b:continue
        mask=arrays[row['id']][1][t-y:b-y,l-x:r-x]
        view=result[t-top:b-top,l-left:r-left];valid=mask!=255
        conflicts[t-top:b-top,l-left:r-left]|=valid&(view!=255)&(view!=mask)
        view[valid]=mask[valid]
    result[conflicts]=255
    return result,int(conflicts.sum())


def primitive_coverage(target,geometry):
    """Unclassified primitive corridor coverage of assigned native ink only."""
    raster=np.zeros(target.shape,np.uint8)
    for candidate in geometry['candidates']:
        if candidate['kind']=='line':points=np.asarray(candidate['points_px'],float)
        else:
            radius=candidate['radius_px'];start=candidate.get('start_angle_deg',0);sweep=candidate.get('sweep_angle_deg',360)
            angles=np.radians(np.linspace(start,start+sweep,min(8192,max(32,int(radius*abs(np.radians(sweep)))))))
            points=np.asarray(candidate['center_px'])+radius*np.c_[np.cos(angles),np.sin(angles)]
        require(np.isfinite(points).all(),'Nonfinite primitive')
        cv2.polylines(raster,[np.rint(points).astype(np.int32)],False,255,1)
    distance=cv2.distanceTransform(255-raster,cv2.DIST_L2,3)
    return {str(c):{'assigned_ink_pixels':int((target==c).sum()),
                   'covered_ink_pixels':int(((target==c)&(distance<=POLICY['primitive_band_px'])).sum())}
            for c in POLICY['geometry_classes']}


def admission(metrics,cv_rows,repeat_equal,resources_pass):
    p=POLICY['necessary_partial_gates']
    ious=[r['IoU_within_review'] for r in metrics['classes'][1:]]
    gates={'each_foreground_IoU':all(v is not None and v>=p['minimum_each_foreground_IoU'] for v in ious),
        'reviewed_background':metrics['background_false_foreground_fraction'] is not None and metrics['background_false_foreground_fraction']<=p['maximum_background_false_foreground'],
        'thin_axis_connected_runs':metrics['thin_axis_connected_80_fraction'] is not None and metrics['thin_axis_connected_80_fraction']>=p['minimum_thin_axis_connected_80_fraction'],
        'arrow_regression':all(r['guided']['matched']>=r['original']['matched'] and r['guided']['false_tips']<=r['original']['false_tips'] for r in cv_rows),
        'primitive_regression':all(sum(v['covered_ink_pixels'] for v in r['guided']['primitive_coverage'].values())>=sum(v['covered_ink_pixels'] for v in r['original']['primitive_coverage'].values()) for r in cv_rows),
        'repeat_equal':repeat_equal,'resources':resources_pass}
    return {'necessary_partial_gates':gates,'necessary_partial_gates_pass':all(gates.values()),
        'production_admitted':False,'physical_geometry_approved':False,
        'missing_acceptance_evidence':['Dense full-view targets and independently confirmed source families.',
            'Validated primitive identities, dimension associations and physical CAD reconstruction.'],
        'full_page_precision':None,'independent_evaluation':False}
