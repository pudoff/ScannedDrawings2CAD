"""Map working-pixel CAD to the prepared image without deforming DXF circles."""
import numpy as np
from .comparison_frame_v1 import svg_to_canvas
from .service_profiles_v6.api import ReviewedComparisons
from .web_correction_contract_v1 import decode


def native_frame(export):
    width,height = export['working_size_px']; m = np.asarray(export['working_to_processed'],float)
    if type(width)is not int or type(height)is not int or min(width,height)<=0 or width*height>4194304:
        raise ValueError('Invalid native CAD frame')
    if m.shape!=(3,3) or not np.isfinite(m).all() or not np.array_equal(m[2],[0,0,1]) or m[0,0]<=0 or m[1,1]<=0:
        raise ValueError('Invalid CAD to prepared-image matrix')
    if m[0,1]!=0 or m[1,0]!=0: raise ValueError('Unsupported native CAD frame shear')
    return [-.5,-.5,width-.5,height-.5],m@np.array([[1,0,0],[0,-1,height-1],[0,0,1]],float)


class NativeComparisons(ReviewedComparisons):
    def _render(self,path,digest,render_box):
        report = decode((path.parent/'report.json').read_bytes()); page = int(path.stem.split('-')[-1])-1
        exported = report['pages'][page].get('demo_export',{})
        if exported.get('mode')=='native_annotations_draft': render_box,_ = native_frame(exported)
        return super()._render(path,digest,render_box)

    def describe(self,run,page):
        value = super().describe(run,page); exported = value['export_report']
        if exported.get('mode')!='native_annotations_draft': return value
        box,mapping = native_frame(exported)
        value['frame'].update(render_box=box,dxf_to_canvas=mapping.tolist(),
            dxf_units='working_pixels_unitless',CAD_entities_rescaled=False)
        if value['dxf']['status']=='ready':
            value['dxf']['svg_to_canvas'] = svg_to_canvas(box,value['dxf']['svg_size'],mapping)
        value['native_CAD_counts'] = {key:exported.get(key,0) for key in
            ('geometry_entity_count','native_text_count','visible_text_count','native_dimensions','automatic_dimensions','review_dimensions')}
        return value
