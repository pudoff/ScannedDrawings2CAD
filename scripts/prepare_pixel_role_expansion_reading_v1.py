"""Literal assistant observations after viewing twelve original native RGB crops.

Coordinates below are visual review decisions, never model-generated targets.
Run once to a new reading file; later corrections require another version.
"""
import json
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cad_recovery.dataset_layout_v1 import encoded


def line(role,points,band=5):return {'role':role,'kind':'polyline','points':points,'band_px':band}
def box(role,b):return {'role':role,'kind':'box','box':b}
def unknown(b,reason):return {'kind':'box','box':b,'reason':reason}
def head(tip,direction):
    norm=math.hypot(*direction)
    return {'state':'visible_head','tip_px':tip,'direction':[v/norm for v in direction],
            'observation':'Visible filled dimension or leader head; tip and direction read from the native raster, no detector output used.'}
def negative(b,observation):return {'state':'negative_region','kind':'box','box':b,'observation':observation}


def main():
    path=ROOT/'configs/reviews/pixel-role-expansion-reading-v5.json'
    if path.exists():raise ValueError('Choose a new reading version')
    reading=json.loads((ROOT/'configs/reviews/pixel-role-expansion-reading-v1.json').read_text('utf-8'))
    reading['patches'].append({'id':'train-extra-1-angle','source_id':'train-extra-1',
        'source_box':[2400,3700,3424,4724],'annotations':[],'unknown_regions':[],
        'background_regions':[],'arrow_controls':[],'observations':[]})
    patches={p['id']:p for p in reading['patches']}
    def assign(name,annotations,unknowns=(),controls=(),note=''):
        p=patches[name];p['annotations']=[{'id':name+'-role-'+str(i),**a} for i,a in enumerate(annotations,1)]
        p['unknown_regions']=[{'id':name+'-unknown-'+str(i),**a} for i,a in enumerate(unknowns,1)]
        p['arrow_controls']=[{'id':name+'-arrow-'+str(i),**a} for i,a in enumerate(controls,1)]
        w,h=p['source_box'][2]-p['source_box'][0],p['source_box'][3]-p['source_box'][1]
        p['background_regions']=[{'id':'reviewed-clear-paper','kind':'box','box':[0,0,w,h],
            'observation':'Inspected clear paper throughout this native crop; only near-white pixels beyond the fixed ink clearance may be background. Unassigned ink remains ignore.'}]
        p['observations']=['Viewed the exact native RGB crop and its coordinate grid; no resize, threshold or detector result was used to infer semantic role.',note]

    assign('train-new-1-1',[
        line('title_grid',[[44,155],[1023,155]],8),line('title_grid',[[44,274],[1023,274]],6),
        line('title_grid',[[44,391],[1023,391]],6),line('title_grid',[[44,509],[1023,509]],8),
        *[line('title_grid',[[x,155],[x,509]],8) for x in (44,209,446,990)],
        box('text',[75,412,193,485]),box('text',[245,412,410,485]),box('text',[590,409,860,486])],
        [unknown([0,620,1024,642],'Paper edge and outer printed margin are not the title-grid role.')],
        [negative([0,140,1024,520],'Change-table grid and text; no dimension head anywhere in this assigned title fragment.')],
        'The crop is the change table of a purchased-items sheet. It contains title-grid strokes and text, no component contour or dimension targets.')
    assign('train-new-1-2',[box('text',[300,145,800,235]),box('text',[215,889,815,991])],
        [unknown([115,0,143,1024],'Purchased-items table column is outside the current title_grid taxonomy.'),
         *[unknown([0,y-7,1024,y+7],'Inventory row ruling is neither component contour nor title-block grid.') for y in (92,282,470,660,848)]],
        [negative([150,120,950,1010],'Inventory standards text and paper; no engineering dimension arrow in this assigned table region.')],
        'The main inventory ruling is explicitly unknown, not mislabelled as title_grid. Only the two isolated standards inscriptions receive text targets.')
    assign('train-new-2-1',[
        line('part_contour',[[65,85],[136,85],[136,370],[65,370],[65,85]],3),
        {'role':'part_contour','kind':'circle','center':[409,228],'radius':143,'band_px':3},
        {'role':'part_contour','kind':'circle','center':[409,228],'radius':91,'band_px':3},
        line('axis_construction',[[30,228],[589,228]],2),line('axis_construction',[[409,44],[409,418]],2),
        line('dimension_line',[[76,430],[125,430]],2),
        line('dimension_line',[[330,324],[485,137]],2),line('dimension_line',[[359,175],[542,350]],2),
        line('extension_line',[[65,375],[65,442]],2),line('extension_line',[[136,375],[136,442]],2),
        box('text',[91,403,122,428]),box('text',[500,36,548,91]),box('text',[555,353,618,414]),
        *[line('hatching',points,2) for points in
          ([[280,190],[366,104]],[[278,211],[393,96]],[[283,244],[315,212]],
           [[425,136],[461,100]],[[444,139],[476,108]],[[500,250],[542,208]],
           [[491,280],[545,226]],[[393,319],[353,359]],[[411,319],[373,357]])]],
        [unknown([55,123,145,145],'Green hidden-edge convention has no class in the present taxonomy.'),
         unknown([55,308,145,330],'Hidden line is not promoted to a visible part boundary.'),
         unknown([620,300,700,520],'CAD crosshair and square marker are source UI residue, not engineering contour.'),
         unknown([60,422,85,438],'Dimension head excluded from shaft masks.'),unknown([120,422,142,438],'Dimension head excluded from shaft masks.'),
         unknown([337,155,363,179],'Inner diameter head excluded from pixel shaft supervision.'),unknown([456,270,478,294],'Inner diameter head excluded.'),
         unknown([313,321,339,344],'Outer diameter head excluded.'),unknown([483,109,505,135],'Outer diameter head excluded.')],
        [head([65,430],[-1,0]),head([136,430],[1,0]),head([346,165],[-1,-1]),head([472,291],[1,1]),
         negative([620,300,700,520],'CAD crosshair and square marker are not dimension arrowheads.')],
        'The original has red dimensions, cyan axes and green hidden lines. The black ring and hatch are identified by geometry and context, not by a color-to-role rule.')
    assign('train-new-2-2',[
        *[line('title_grid',[[x,10],[x,338]],3) for x in (133,200,221,281,697,994)],
        *[line('title_grid',[[0,y],[281,y]],2) for y in (40,70,129,159,189,219,249,278,308,338)],
        *[line('title_grid',[[281,y],[994,y]],2) for y in (10,100,249,338)],
        line('title_grid',[[697,219],[994,219]],2),
        *[line('title_grid',[[x,100],[x,219]],2) for x in (726,755,785,838,887)],
        box('text',[514,24,747,83]),box('text',[355,113,614,187]),box('text',[442,205,525,241]),
        box('text',[406,275,558,308]),box('text',[695,222,991,247])],
        [unknown([0,330,1024,360],'Outer copier/footer inscriptions are not title-grid boundaries.')],
        [negative([290,110,680,320],'Title text including slanted and red glyphs; no dimension head in this assigned cell.')],
        'Title-block cell boundaries and printed words were inspected separately. The saved red text is retained exactly and does not define a role by color alone.')
    assign('train-extra-1-1',[
        line('extension_line',[[317,120],[317,1018]],6),
        line('dimension_line',[[395,168],[1018,168]],6),line('dimension_line',[[395,392],[1005,392]],6),
        box('text',[662,14,865,153]),box('text',[498,239,779,377])],
        [unknown([308,151,392,185],'Large filled arrowhead excluded from shaft-only labels.'),
         unknown([308,375,392,409],'Large filled arrowhead excluded from shaft-only labels.'),
         unknown([805,630,1024,1024],'Partial roughness symbol and incomplete contour stay unknown.')],
        [head([317,168],[-1,0]),head([317,392],[-1,0]),
         negative([495,235,785,380],'Printed 33 with star is text, with no filled dimension head in this assigned region.')],
        'Monochrome native raster: two large filled heads on a shared extension, dimension shafts, and 38 / 33-star text. No physical value or scale is approved.')
    assign('train-extra-1-2',[
        line('part_contour',[[0,269],[503,269]],6),
        line('part_contour',[[504,269],[531,295],[559,316],[590,333],[626,344],[674,352],[1018,352]],6),
        line('part_contour',[[0,610],[1018,610]],6),line('part_contour',[[0,681],[1018,681]],6),
        line('extension_line',[[593,375],[593,1018]],5),
        line('leader_line',[[274,565],[481,565],[553,403]],5),
        box('text',[270,411,503,551]),box('text',[258,891,451,1020]),
        *[line('hatching',points,4) for points in
          ([[135,271],[126,280]],[[116,399],[243,272]],[[110,500],[339,271]],
           [[326,395],[451,270]],[[429,395],[528,296]],[[670,367],[506,531]],
           [[797,354],[606,545]],[[898,354],[655,597]],[[1000,354],[755,600]],
           [[340,625],[383,668]],[[443,625],[486,668]],[[551,625],[594,668]],[[657,625],[700,668]])]],
        [unknown([545,329,609,406],'Leader head and its contour junction are excluded from pixel shaft labels.'),
         unknown([26,679,69,758],'Partly visible lower dimension head is not a full control target.'),
         unknown([815,493,850,1024],'Dot-ended leader has no reviewed endpoint convention in the current mask dictionary.'),
         unknown([230,0,1024,260],'Partial roughness annotation and adjacent outlines are not assigned in this crop.'),
         {'kind':'polyline','points':[[120,285],[110,367],[106,449],[112,505],[131,556],[168,610],[217,663],[280,716],[355,762],[443,803],[550,839],[671,866],[805,882],[1018,881]],'band_px':12,
          'reason':'Enlarged-detail framing circle is a view convention, not a physical part contour.'}],
        [head([590,349],[.48,-.88]),
         negative([675,390,1000,595],'Diagonal hatching and a dot-ended leader are visible; the dot is not a filled triangular arrowhead in this assigned negative area.')],
        'Monochrome section contains hatching, a curved contour, R2 leader and intersections. Hatch corridors are selected individually; overprinted text and endpoint conventions remain explicit unknowns.')
    assign('train-extra-2-1',[
        line('part_contour',[[0,237],[92,237],[264,408],[452,408],[452,620],[264,620],[264,408]],6),
        line('part_contour',[[263,620],[358,674],[451,620]],6),
        line('part_contour',[[170,595],[170,738],[198,756],[516,756],[546,738],[546,594]],6),
        line('extension_line',[[181,45],[181,225]],5),line('extension_line',[[534,45],[534,371]],5),
        line('part_contour',[[181,437],[181,536]],5),line('part_contour',[[534,439],[534,536]],5),
        line('extension_line',[[264,157],[264,374]],5),line('extension_line',[[452,157],[452,374]],5),
        line('dimension_line',[[340,204],[375,204]],5),line('dimension_line',[[979,246],[979,304]],5),
        line('dimension_line',[[979,493],[979,592],[600,592]],5),
        box('text',[588,45,1023,189]),box('text',[584,435,974,579]),
        *[line('hatching',points,4) for points in
          ([[194,446],[244,396]],[[191,535],[253,473]],[[192,625],[252,565]],
           [[196,713],[272,637]],[[255,742],[314,683]],[[342,742],[417,667]],
           [[435,742],[526,651]],[[463,600],[519,544]],[[462,506],[516,452]])]],
        [unknown([180,0,536,40],'Top diameter heads are clipped by the crop and excluded.'),
         unknown([253,186,338,222],'Left filled dimension head excluded from shaft labels.'),
         unknown([377,186,461,222],'Right filled dimension head excluded from shaft labels.'),
         unknown([956,305,999,488],'Two chamfer dimension heads excluded from pixel shaft labels.'),
         unknown([0,780,1024,1024],'Roughness symbols and incomplete contours below this section remain unknown.')],
        [head([263,204],[-1,0]),head([452,204],[1,0]),head([978,383],[0,1]),head([978,408],[0,-1]),
         negative([589,435,974,580],'Printed chamfer annotation is text, not a filled arrowhead.')],
        'Monochrome nut section: independent hatch strokes, stepped contour and dimension extensions. Chamfer heads and roughness are excluded; the chamfer text remains literal glyph ink.')
    assign('train-extra-2-2',[
        line('axis_construction',[[4,460],[1018,460]],5),line('axis_construction',[[532,0],[532,1018]],5),
        line('dimension_line',[[280,0],[857,1018]],5)],
        [unknown([0,585,470,945],'Cutting-plane marker and letter A are not ordinary dimension heads or part contours.')],
        [negative([470,350,660,540],'Axis crossing and diagonal diameter shaft without any head; hard intersection negative.')],
        'The dashed axes and oblique diameter shaft cross. Both roles are retained and their intersection is buffered to ignore; cutting-plane arrow conventions remain outside this control.')
    assign('train-extra-1-angle',[
        line('part_contour',[[0,256],[821,256],[835,244],[855,239],[919,239],[932,244],[941,257],[941,291],[1018,291]],5),
        line('part_contour',[[165,100],[165,230],[527,438]],5),
        line('extension_line',[[941,310],[941,1010]],4),
        line('dimension_line',[[0,919],[860,919]],4),
        line('dimension_line',[[521,327],[520,331],[519,335],[518,339],[516,343]],4),
        line('leader_line',[[530,360],[637,530],[875,530]],4),
        box('text',[635,398,890,526])],
        [unknown([508,250,540,328],'Upper filled head on the angular dimension is excluded from shaft targets.'),
         unknown([483,340,531,418],'Lower filled head on the angular dimension is excluded from shaft targets.'),
         unknown([89,133,285,166],'Two straight width heads at the upper crop edge are not assigned as pixel shafts.'),
         unknown([866,901,946,937],'Bottom straight dimension head is excluded from shaft targets.'),
         unknown([180,370,447,655],'Roughness symbol and its glyphs remain outside this partial target review.')],
        [head([530,257],[.125,-.992]),head([487,414],[-.291,.957]),
         negative([630,393,900,548],'Literal angle text 30 degrees and its underline; no filled head in this assigned text region.'),
         negative([180,440,447,660],'Roughness symbol and the text 10 are not filled dimension arrowheads.')],
        'Additional monochrome angular dimension on the same train source: two visible heads meet a short curved shaft. Tips follow the observed head axes; no complete hidden arc or physical 30-degree target is inferred.')
    assign('validation-new-1-1',[
        {'role':'part_contour','kind':'circle','center':[297,391],'radius':177,'band_px':7},
        line('axis_construction',[[100,391],[349,391]],5),line('axis_construction',[[297,156],[297,780]],5),
        line('axis_construction',[[100,897],[349,897]],5),line('axis_construction',[[297,840],[297,1018]],5),
        line('extension_line',[[481,391],[1018,391]],4),line('extension_line',[[448,897],[1018,897]],4),
        line('leader_line',[[500,204],[724,0]],5),line('leader_line',[[43,619],[93,575]],5),
        line('leader_line',[[508,797],[790,677],[960,677]],5),box('text',[798,562,974,661])],
        [unknown([96,503,181,579],'Filled diameter head excluded from pixel shaft labels.'),
         unknown([425,202,502,281],'Filled diameter head excluded from pixel shaft labels.'),
         unknown([402,791,513,850],'Lower diameter head excluded.'),
         unknown([80,932,177,1004],'Lower diameter head excluded.'),
         unknown([145,743,449,1024],'Lower clipped circle and axis intersections require a separate complete contour review.')],
        [head([425,271],[-.75,.66]),head([166,507],[.75,-.66]),head([427,839],[-.91,.41]),
         negative([200,330,390,451],'Circle-center cross and axis intersection contains no arrowhead.')],
        'Monochrome diagnostic validation: hole outline, thin axes and leader shafts. Coincident extension and axis roles are separated at their visible purposes; lower clipped circle stays unknown.')
    assign('validation-new-1-2',[
        line('part_contour',[[184,444],[189,392],[207,345],[237,305],[277,270],[324,244],[371,230],[398,227],[438,231],[482,245],[524,270],[560,304],[585,345],[601,392],[608,439],[608,951]],7),
        line('part_contour',[[184,442],[184,951]],7),
        line('axis_construction',[[397,0],[397,1018]],5),line('axis_construction',[[136,439],[397,439]],4),
        line('axis_construction',[[149,956],[397,956]],4),
        line('extension_line',[[397,439],[1018,439]],4),line('extension_line',[[397,956],[1018,956]],4),
        line('dimension_line',[[272,617],[515,617]],5),line('dimension_line',[[618,617],[817,617]],5),
        line('leader_line',[[0,100],[129,100],[205,194]],5),
        box('text',[7,0,123,93]),box('text',[703,509,815,604])],
        [unknown([180,601,269,635],'Left width head excluded from shaft labels.'),unknown([521,601,613,635],'Right width head excluded.'),
         unknown([199,192,274,284],'R9 leader head excluded from pixel shaft labels.'),
         unknown([125,949,666,1024],'Lower slot end is clipped; its precise arc target is not released.')],
        [head([185,617],[-1,0]),head([607,617],[1,0]),head([265,272],[.64,.77]),
         negative([320,355,470,500],'Thin center axes and extension junction are not filled heads.')],
        'Slot end is visibly curved with vertical sides, a radius leader, width heads and a central thin axis. No hidden complete circle or physical radius target is inferred.')
    assign('validation-new-2-1',[
        line('part_contour',[[168,1018],[168,236],[173,211],[184,189],[202,173],[222,163],[244,160],[1018,160]],7),
        {'role':'part_contour','kind':'circle','center':[256,249],'radius':20,'band_px':6},
        line('axis_construction',[[0,249],[1018,249]],4),line('axis_construction',[[256,211],[256,1018]],4),
        line('leader_line',[[202,424],[220,365]],4),line('leader_line',[[300,113],[337,0]],4)],
        [unknown([246,145,304,234],'Upper tiny-hole head and adjacent contour junction excluded.'),
         unknown([211,273,256,354],'Lower tiny-hole head excluded from shaft masks.')],
        [head([266,226],[-.3,.954]),head([249,273],[.3,-.954]),
         negative([450,210,1000,310],'Only a thin dashed horizontal axis is visible in this assigned strip.')],
        'A rounded corner, tiny circular hole and interrupted thin axes are distinct in the original. The two close leader heads are separate directed observations, not two line endpoints.')
    assign('validation-new-2-2',[
        line('axis_construction',[[10,757],[487,757]],4),line('axis_construction',[[653,757],[1018,757]],4),
        line('part_contour',[[0,845],[485,845]],7),line('part_contour',[[674,845],[1018,845]],7)],
        [unknown([474,0,689,860],'Wavy break edges are a conventional interruption, not a physical arc or visible part boundary.'),
         unknown([401,951,1024,1024],'Dimension text and a cropped shaft enter below the reviewed geometry area.')],
        [negative([465,0,697,865],'Two wavy break conventions crossing the shared axis; no arrowheads and no physical open circular arc.')],
        'The interrupted part and horizontal thin axis were inspected. The wavy break convention remains unknown with a buffer; its blank gap is never bridged into a physical contour.')

    definitions={
      'train-new-1':('purchased_items_list', [('parts_list',[400,100,9820,6300]),('title_block',[5400,6500,9820,6910])]),
      'train-new-2':('component_drawing',[('drawing_view',[420,490,1010,940]),('title_block',[240,1440,1345,1780]),('notes',[420,270,1180,430])]),
      'train-extra-1':('component_drawing',[('drawing_view',[1000,1300,4700,4700]),('drawing_view',[2500,5500,7400,9150]),('title_block',[9500,8450,13900,9820]),('notes',[9500,2800,13900,6000])]),
      'train-extra-2':('component_drawing',[('drawing_view',[650,1350,2700,4750]),('drawing_view',[2800,1350,5600,4450]),('drawing_view',[800,4900,2700,6800]),('title_block',[5400,5550,9810,6900]),('notes',[5500,800,9800,2150])]),
      'validation-new-1':('component_drawing',[('drawing_view',[900,650,3420,6000]),('drawing_view',[4280,650,4930,5800]),('title_block',[5400,5550,9810,6900]),('notes',[5350,4400,8100,4920])]),
      'validation-new-2':('component_drawing',[('drawing_view',[650,1200,4900,4150]),('drawing_view',[550,4490,5200,5830]),('title_block',[5400,5550,9810,6900]),('notes',[5350,4460,9800,5000])])}
    reading['source_reviews']=[]
    for source_id,(kind,regions) in definitions.items():
        reading['source_reviews'].append({'source_id':source_id,'drawing_kind':kind,
            'observations':['Viewed the pinned overview to locate the complete drawing views, title block and notes before assigning native crop roles.',
                'Page boxes are coarse visual proposals for later region review; they are not dense masks, exact geometry, or admitted detector targets.'],
            'regions':[{'id':source_id+'-region-'+str(i),'role':role,'kind':'box','box':b,
                        'observation':'Coarse region from the original page overview; includes surrounding dimension or note context and has no physical coordinate authority.'} for i,(role,b) in enumerate(regions,1)]})
    reading['limitations']=[
        'Assistant visual review only, no human verification or physical units/scale/coordinate targets.',
        'Partial ink islands from reviewed corridors; unassigned ink, overlapping roles and conventions stay ignore.',
        'Two added monochrome train sources share an existing conservative group: source count is not independent-family count.',
        'All validation is diagnostic, forbidden for training and checkpoint or model selection.',
        'The inventory main table has no part geometry; its ruling is outside title_grid and remains unknown.',
        'Color and luminance shift remain even after grayscale; derived contrast statistics are not a new domain guarantee.',
        'Coarse page regions are proposals only; complete-view dense segmentation and real scan domain acceptance remain open.',
        'Arrow controls are assigned visible tips/negative ROIs, not exhaustive precision or automated training admission.',
        'No new detector, OCR, VLM, GPU inference or training is executed for this release.']
    path.write_bytes(encoded(reading));print(path.relative_to(ROOT))


if __name__=='__main__':main()
