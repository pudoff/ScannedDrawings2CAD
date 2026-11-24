"""Explicit pixel-centre frames for source/observations/exported DXF comparison."""
import numpy as np
from .observation_graph_v1 import matrix


def frame(page,graph,source_size,inspection=None,rotation=0):
    sw,sh=source_size
    if any(type(v)is not int or not 0<v<=32768 for v in source_size):raise ValueError('Invalid source preview size')
    prep=page.get('preprocessing')
    native=inspection and inspection.get('source_coordinate_chain')
    if native:
        cw,ch=inspection['preview']['size_px'];analysis=native.get('analysis_size_px',inspection['source_size_px'])
        scale=np.array([[cw/analysis[0],0,(cw/analysis[0]-1)/2],[0,ch/analysis[1],(ch/analysis[1]-1)/2],[0,0,1]])
        return {'canvas_size_px':[cw,ch],'source_to_canvas':np.eye(3).tolist(),
            'observation_to_canvas':(scale@matrix(native.get('original_to_analysis',np.eye(3)))).tolist(),
            'source_label':'Навигационный исходник','dxf_to_canvas':None,'render_box':None}
    if prep:
        if prep['input_size_px']!=source_size:raise ValueError('Source preview/preprocessing frame mismatch')
        cw,ch=prep['processed_size_px'];source=matrix(prep['transforms']['input_to_processed']);obs=np.eye(3)
        if any(type(v)is not int or not 0<v<=32768 for v in (cw,ch)):raise ValueError('Invalid processed canvas size')
    else:cw,ch=sw,sh;source=np.eye(3);obs=np.eye(3)
    kind=graph['page']['kind']
    if kind=='raster':
        if not prep:raise LookupError('Raster export frame is not recorded')
        dxf=np.array([[1,0,0],[0,-1,ch-1],[0,0,1]],float)
        box=[-.5,-.5,cw-.5,ch-.5]
    elif kind=='vector':
        p=graph['page'];w,h=float(p['width_mm']),float(p['height_mm'])
        factor=(p.get('model_per_paper') or 1.) if graph.get('coordinate_space')=='paper_mm' else 1.
        if not np.isfinite([w,h,factor]).all() or min(w,h,factor)<=0:raise ValueError('Invalid PDF frame')
        if rotation not in [0,90,180,270]:raise ValueError('Unsupported PDF rotation')
        a=np.deg2rad(-rotation);rotate=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        model_to_source=np.array([[sw/(w*factor),0,-.5],[0,-sh/(h*factor),sh-.5],[0,0,1]])@np.linalg.inv(rotate)
        dxf=source@model_to_source
        corners=(rotate@np.array([[0,0,1],[w*factor,0,1],[w*factor,h*factor,1],[0,h*factor,1]]).T).T[:,:2]
        box=[*corners.min(axis=0),*corners.max(axis=0)]
    else:raise LookupError('Unverified export coordinate frame')
    return {'canvas_size_px':[cw,ch],'source_to_canvas':source.tolist(),'observation_to_canvas':obs.tolist(),
        'source_label':'Исходный лист','dxf_to_canvas':matrix(dxf).tolist(),'render_box':list(map(float,box))}


def svg_to_canvas(render_box,svg_size,dxf_to_canvas):
    l,b,r,t=render_box;w,h=svg_size
    scale=min(w/(r-l),h/(t-b));px=(w-(r-l)*scale)/2;py=(h-(t-b)*scale)/2
    model_to_svg=np.array([[scale,0,px-l*scale],[0,-scale,py+t*scale],[0,0,1]])
    return (matrix(dxf_to_canvas)@np.linalg.inv(model_to_svg)).tolist()
