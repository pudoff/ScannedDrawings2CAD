// Public, DOM-independent delta protocol. Existing V18 assets are immutable.
const key = row => JSON.stringify([row.region_id, row.cell_id]);
const sameArray = (a,b) => Array.isArray(a)&&Array.isArray(b)&&a.length===b.length&&a.every((v,i)=>v===b[i]);
const sameCell = (a,b) => !!a&&a.region_id===b.region_id&&a.cell_id===b.cell_id&&a.origin===b.origin&&sameArray(a.handles,b.handles);
const sameArea = (a,b) => !!a&&a.handle===b.handle&&sameArray(a.box_px,b.box_px);

export function textDelta(previous, current, removals={}) {
  const priorCells=new Map(previous.selections.map(v=>[key(v),v]));
  const priorAreas=new Map(previous.associations.map(v=>[v.handle,v]));
  const cells=current.selections,areas=current.associations;
  if(new Set(cells.map(key)).size!==cells.length||new Set(areas.map(v=>v.handle)).size!==areas.length)
    throw new Error('duplicate_text_choice');
  const clearCells=removals.clear_cells||[],clearAreas=removals.clear_associations||[];
  const clearing=new Set(clearCells.map(key));
  if(clearing.size!==clearCells.length||new Set(clearAreas).size!==clearAreas.length||
      cells.some(v=>clearing.has(key(v)))||areas.some(v=>clearAreas.includes(v.handle))||
      clearCells.some(v=>!priorCells.has(key(v)))||clearAreas.some(v=>!priorAreas.has(v)))
    throw new Error('invalid_text_removal');
  return {selections:cells.filter(v=>!sameCell(priorCells.get(key(v)),v)),
    associations:areas.filter(v=>!sameArea(priorAreas.get(v.handle),v)),
    clear_cells:clearCells,clear_associations:clearAreas};
}

export function boundedTextBody(reference, delta, reviewer, reviewed) {
  if(reviewed!==true||typeof reviewer!=='string'||!reviewer.trim()||reviewer.length>120)
    throw new Error('reviewer_and_confirmation_required');
  const allowed=['proposal_id','preview_sha256','review_sha256','source_kind'];
  if(Object.keys(reference).some(k=>!allowed.includes(k)))throw new Error('unknown_text_reference');
  if(!/^[0-9a-f-]{36}$/i.test(reference.proposal_id)||
      !/^[0-9a-f]{64}$/.test(reference.preview_sha256)||!/^[0-9a-f]{64}$/.test(reference.review_sha256)||
      !['proposal','published'].includes(reference.source_kind))throw new Error('invalid_text_reference');
  const keys=['selections','associations','clear_cells','clear_associations'];
  if(Object.keys(delta).some(k=>!keys.includes(k))||keys.some(k=>!Array.isArray(delta[k])))
    throw new Error('invalid_text_delta');
  if(!delta.selections.length&&!delta.associations.length&&!delta.clear_cells.length&&!delta.clear_associations.length)
    throw new Error('text_choice_unchanged');
  const result=JSON.stringify({...reference,...delta,reviewer:reviewer.trim(),reviewed:true});
  if(new TextEncoder().encode(result).length>4096)throw new Error('text_body_limit');
  return result;
}
