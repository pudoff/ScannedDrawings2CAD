"""Bounded homogeneous fits and source-contiguous segmentation; no raster edits."""
import math
from copy import deepcopy
import numpy as np
from .anchored_curve_fit_v1 import fit as old_fit,circle_solution
from .cad_object_assembly_v1 import _reverse_error
from .curve_graph_v1 import native_support
from .stroke_curve_control_v1 import distance_to_polyline

POLICY={'version':'cad-primitive-fit/2.0','maximum_fit_calls':8192,'maximum_depth':12,
        'maximum_points_per_fit':8192,'maximum_curve_samples':120000,
        'maximum_families':256,'maximum_family_pairs':2048,
        'p95_px':1.25,'max_px':2.,'sensitivity_px':2.,'minimum_length_px':8.,
        'minimum_sagitta_px':1.,'family_parameter_ambiguity_px':2.,
        'maximum_refine_positions':9,'tangent_tolerance_deg':8.}


def checked(candidate,points,support):
    if candidate is None:return None
    if candidate['kind'] in ('arc','circle'):
        sweep=math.radians(candidate.get('sweep_angle_deg',360))
        if candidate['radius_px']*sweep/.5>POLICY['maximum_curve_samples']:return None
    reverse=_reverse_error(candidate,points)
    if reverse is None or reverse>POLICY['max_px']+math.sqrt(.5):return None
    if not native_support(candidate,support):return None
    return {**candidate,'reverse_source_pixel_distance_px':reverse}


def sector(points,center,radius):
    angles=np.unwrap(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))
    delta=float(angles[-1]-angles[0]);sweep=abs(delta)
    # A sector cannot double back or jump to a missing complementary sector.
    if not 0<sweep<2*math.pi-1e-7:return None
    if np.max(abs(np.diff(angles)),initial=0)>math.pi/2:return None
    progress=np.diff(angles)*np.sign(delta)
    if np.sum(np.maximum(-progress,0))>math.sqrt(.5)/max(radius,1)*2:return None
    start=float(angles[0] if delta>0 else angles[-1])
    error=abs(np.linalg.norm(points-center,axis=1)-radius)
    if np.percentile(error,95)>POLICY['p95_px'] or error.max()>POLICY['max_px']:return None
    return {'kind':'arc','center_px':center.tolist(),'radius_px':float(radius),
            'start_angle_deg':math.degrees(start)%360,'sweep_angle_deg':math.degrees(sweep),
            'fit_max_px':float(error.max()),'fit_p95_px':float(np.percentile(error,95)),
            'fit_rmse_px':float(np.sqrt(np.mean(error**2))),'observed_sweep_angle_deg':math.degrees(sweep)}


class Fitter:
    def __init__(self,support):
        self.support=support;self.calls=0;self.incomplete=False;self.family_pairs=0

    def fit(self,p,closed=False):
        if self.calls>=POLICY['maximum_fit_calls']:
            self.incomplete=True;return None
        self.calls+=1
        if len(p)>POLICY['maximum_points_per_fit'] or len(p)<3:return None
        length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
        if length<POLICY['minimum_length_px']:return None
        found=old_fit(p,closed,self.support,conditioned=True,native_scale=1.)
        # Retain the existing homogeneous circle decision; sector proposals below
        # need a separate stability proof before they can become emitted ARC.
        if found is not None and found['kind']=='circle':return checked(found,p,self.support)
        solution=circle_solution(p,closed,max(self.support.shape)*4)
        proposal=None
        if solution is not None and not closed:
            center,radius,sensitivity=solution
            arc=sector(p,center,radius)
            if arc is not None:
                sweep=math.radians(arc['sweep_angle_deg'])
                sagitta=radius*(1-math.cos(min(sweep,math.pi)/2))
                line_error=float(np.sqrt(np.mean(distance_to_polyline(p,p[[0,-1]])**2)))
                if sagitta>=POLICY['minimum_sagitta_px'] and arc['fit_rmse_px']<=.65*max(line_error,1e-12):
                    arc.update(parameter_sensitivity_native_px=float(sensitivity),
                               locally_stable=sensitivity<=POLICY['sensitivity_px'])
                    proposal=checked(arc,p,self.support)
        if proposal is not None:return proposal
        if found is not None:return checked(found,p,self.support)
        return None

    def split(self,p,closed=False,depth=0):
        typed=self.fit(p,closed)
        if typed is not None:return [{'points':p,'model':typed}]
        if len(p)<7 or depth>=POLICY['maximum_depth'] or self.incomplete:
            return [{'points':p,'model':None}]
        a,b=p[[0,-1]]
        error=np.linalg.norm(p-a,axis=1) if np.array_equal(a,b) else distance_to_polyline(p,[a,b])
        # Keep enough source edges on each side; do not create zero-length objects.
        cut=int(np.argmax(error[2:-2]))+2
        if error[cut]<=POLICY['max_px'] and not closed:return [{'points':p,'model':None}]
        return self.split(p[:cut+1],False,depth+1)+self.split(p[cut:],False,depth+1)

    def merge(self,parts):
        for _ in range(12):
            changed=False;out=[];i=0
            while i<len(parts):
                if i+1<len(parts):
                    a,b=parts[i:i+2];ma,mb=a['model'],b['model']
                    # A confirmed model transition is not hidden by a coarse fit.
                    weak_line=lambda m:m and m['kind']=='line' and m['fit_max_px']>math.sqrt(.5)
                    compatible=(ma is None or mb is None or ma['kind']==mb['kind'] or weak_line(ma) or weak_line(mb))
                    if compatible and np.array_equal(a['points'][-1],b['points'][0]):
                        p=np.r_[a['points'],b['points'][1:]];candidate=self.fit(p,np.array_equal(p[0],p[-1]))
                        if candidate is not None:
                            out.append({'points':p,'model':candidate});i+=2;changed=True;continue
                out.append(parts[i]);i+=1
            parts=out
            if not changed or self.incomplete:break
        return parts

    def refine_tangent(self,parts):
        """Use independent straight context to locate a raster-ambiguous G1 cut."""
        for index in range(len(parts)):
            middle=parts[index];model=middle['model']
            if not model or model['kind']!='arc':continue
            left=parts[index-1] if index else None;right=parts[index+1] if index+1<len(parts) else None
            left=left if left and left['model'] and left['model']['kind']=='line' else None
            right=right if right and right['model'] and right['model']['kind']=='line' else None
            if left is None and right is None:continue
            p=middle['points'];offset=0;last=len(p)-1
            if left is not None:offset=len(left['points'])-1;p=np.r_[left['points'],p[1:]];last+=offset
            if right is not None:p=np.r_[p,right['points'][1:]]
            extent=math.sqrt(2*model['radius_px']*POLICY['max_px']+POLICY['max_px']**2)
            def positions(cut,begin,end):
                choices=np.flatnonzero(np.linalg.norm(p-p[cut],axis=1)<=extent)
                choices=choices[(choices>=begin)&(choices<=end)]
                if not len(choices):return [cut]
                return sorted(set([cut]+choices[np.linspace(0,len(choices)-1,min(len(choices),POLICY['maximum_refine_positions'])).astype(int)].tolist()))
            lo=positions(offset,2,last-3) if left is not None else [0]
            hi=positions(last,offset+3,len(p)-3) if right is not None else [len(p)-1]
            best=None
            for a in lo:
                for b in hi:
                    if b-a<3 or self.incomplete:continue
                    q=p[a:b+1];lines=[];line_error=0.;valid=True
                    for group,endpoint in [(p[:a+1],0)] if left is not None else []:
                        errors=distance_to_polyline(group,group[[0,-1]])
                        if np.linalg.norm(group[-1]-group[0])<8 or errors.max()>2 or np.percentile(errors,95)>1.25:valid=False;break
                        lines.append((group,endpoint));line_error+=float(np.sum(errors**2))
                    if right is not None:
                        group=p[b:];errors=distance_to_polyline(group,group[[0,-1]])
                        if np.linalg.norm(group[-1]-group[0])<8 or errors.max()>2 or np.percentile(errors,95)>1.25:valid=False
                        lines.append((group,-1));line_error+=float(np.sum(errors**2))
                    if not valid:continue
                    for group,end in lines:
                        if self.calls>=POLICY['maximum_fit_calls']:self.incomplete=True;break
                        self.calls+=1
                        tangent=group[-1].astype(float)-group[0];length=float(np.linalg.norm(tangent));tangent/=length
                        normal=np.array([-tangent[1],tangent[0]])
                        anchor=q[0 if end==0 else -1].astype(float);other=q[-1 if end==0 else 0].astype(float)
                        d=other-anchor;denom=2*float(d@normal)
                        if abs(denom)<1e-8:continue
                        signed=float(d@d)/denom;center=anchor+signed*normal;radius=abs(signed)
                        if not 2<=radius<=max(self.support.shape)*4:continue
                        candidate=sector(q,center,radius)
                        if candidate is None:continue
                        n=q-center;dist=np.linalg.norm(n,axis=1)
                        jac=(center-q)@normal/np.maximum(dist,1e-12)-np.sign(signed)
                        sensitivity=math.hypot(.5/max(float(np.sqrt(np.mean(jac**2))),1e-12),radius*.5/length)
                        if sensitivity>2:continue
                        # Every available straight context must corroborate tangency.
                        for g,e in lines:
                            t=g[-1].astype(float)-g[0];t/=np.linalg.norm(t)
                            radial=q[0 if e==0 else -1]-center;radial/=np.linalg.norm(radial)
                            if abs(float(t@radial))>math.sin(math.radians(POLICY['tangent_tolerance_deg'])):break
                        else:
                            if not checked(candidate,q,self.support):continue
                            score=float(np.sum((dist-radius)**2))+line_error
                            if best is None or score<best[0]:best=(score,a,b,candidate,sensitivity)
            if best is not None:
                _,a,b,candidate,sensitivity=best
                middle.update(points=p[a:b+1],model={**candidate,'locally_stable':True,
                    'parameter_sensitivity_native_px':sensitivity,'parameter_evidence':'observed_straight_tangent_context'})
                for item,q in [(left,p[:a+1]),(right,p[b:])]:
                    if item is None:continue
                    errors=distance_to_polyline(q,q[[0,-1]])
                    item.update(points=q,model={'kind':'line','points_px':q[[0,-1]].astype(float).tolist(),
                        'fit_max_px':float(errors.max()),'fit_p95_px':float(np.percentile(errors,95))})
        return parts

    def refine_curve_change(self,parts):
        """Replace an unstable transition fragment by a supported two-arc cut."""
        output=[];index=0
        while index<len(parts):
            best=None
            if index+2<len(parts):
                a,b,c=parts[index:index+3]
                if all(x['model'] and x['model']['kind']=='arc' for x in (a,b,c)) and not b['model'].get('locally_stable',True):
                    p=np.r_[a['points'],b['points'][1:],c['points'][1:]]
                    first=len(a['points'])-1;last=first+len(b['points'])-1
                    for cut in range(first,last+1):
                        if self.incomplete:break
                        left=self.fit(p[:cut+1]);right=self.fit(p[cut:])
                        if not all(m and m['kind']=='arc' and m.get('locally_stable',True) for m in (left,right)):continue
                        score=left['fit_rmse_px']**2*(cut+1)+right['fit_rmse_px']**2*(len(p)-cut)
                        if best is None or score<best[0]:best=(score,p,cut,left,right)
            if best is not None:
                _,p,cut,left,right=best;output.extend([{'points':p[:cut+1],'model':left},{'points':p[cut:],'model':right}]);index+=3
            else:output.append(parts[index]);index+=1
        return output

    def families(self,items):
        indices=[i for i,o in enumerate(items) if o.get('model') and o['model']['kind']=='arc']
        if len(indices)>POLICY['maximum_families']:
            self.incomplete=True
            return {'status':'candidate_budget','groups':[]}
        proofs={}
        for pos,i in enumerate(indices):
            a=items[i];ma=a['model']
            for j in indices[pos+1:]:
                b=items[j];mb=b['model']
                if len(a['roles']|b['roles'])>1:continue
                uncertainty=ma.get('parameter_sensitivity_native_px',2)+mb.get('parameter_sensitivity_native_px',2)+2
                if (np.linalg.norm(np.asarray(ma['center_px'])-mb['center_px'])>uncertainty
                        or abs(ma['radius_px']-mb['radius_px'])>uncertainty):continue
                self.family_pairs+=1
                if self.family_pairs>POLICY['maximum_family_pairs']:
                    self.incomplete=True
                    return {'status':'pair_budget','groups':[]}
                if self.calls>=POLICY['maximum_fit_calls']:self.incomplete=True;return {'status':'fit_budget','groups':[]}
                self.calls+=1
                points=np.r_[a['points'],b['points']]
                solution=circle_solution(points,True,max(self.support.shape)*4)
                # Two short sectors can localize a hypothesis without making it
                # stable. Check stability after collecting all agreeing sectors.
                if solution is None:continue
                center,radius,sensitivity=solution;members=[]
                for k in indices:
                    if len(a['roles']|items[k]['roles'])>1:continue
                    fitted=sector(items[k]['points'],center,radius)
                    if fitted and checked(fitted,items[k]['points'],self.support):members.append(k)
                if i not in members or j not in members:continue
                combined=np.concatenate([items[k]['points'] for k in members])
                refined=circle_solution(combined,True,max(self.support.shape)*4)
                if refined is None or refined[2]>POLICY['sensitivity_px']:continue
                center,radius,sensitivity=refined
                fitted=[checked(sector(items[k]['points'],center,radius),items[k]['points'],self.support) for k in members]
                if not all(fitted):continue
                key=tuple(members)
                proofs.setdefault(key,{'members':members,'center_px':center.tolist(),'radius_px':float(radius),
                    'parameter_sensitivity_native_px':float(sensitivity),'models':fitted})
        groups=[]
        for i in indices:
            options=[g for g in proofs.values() if i in g['members']]
            if not options:continue
            best=max(options,key=lambda g:len(g['members']))
            if any(np.linalg.norm(np.asarray(g['center_px'])-best['center_px'])>2 or abs(g['radius_px']-best['radius_px'])>2
                   for g in options):continue
            local=best['members'].index(i)
            items[i]['model']={**best['models'][local],'locally_stable':True,
                'circle_family_support':best['members'],'parameter_sensitivity_native_px':best['parameter_sensitivity_native_px']}
            if best['members'] not in [g['members'] for g in groups]:groups.append({k:v for k,v in best.items() if k!='models'})
        # A stable independent family may corroborate a short source sector
        # that could not produce any stable local circle hypothesis of its own.
        for item in items:
            model=item.get('model')
            if model and model['kind'] not in ('line','arc'):continue
            if model and model['kind']=='arc':continue
            p=item['points']
            if len(p)<3:continue
            line_error=distance_to_polyline(p,p[[0,-1]])
            if line_error.max()<=math.sqrt(.5):continue
            matches=[]
            for group in groups:
                if len(item['roles']|items[group['members'][0]]['roles'])>1:continue
                candidate=checked(sector(p,np.asarray(group['center_px']),group['radius_px']),p,self.support)
                if candidate is not None:matches.append((group,candidate))
            if len(matches)==1:
                group,candidate=matches[0]
                item['model']={**candidate,'locally_stable':True,'circle_family_support':group['members'],
                    'parameter_sensitivity_native_px':group['parameter_sensitivity_native_px']}
        return {'status':'evaluated','groups':groups,'pairs':self.family_pairs}
