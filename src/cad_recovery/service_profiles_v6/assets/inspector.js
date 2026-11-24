const NS = 'http://www.w3.org/2000/svg';
const labels = {geometry:'Геометрия', text:'Текст / знак', arrow:'Стрелка', association:'Размерная связь', endpoint:'Конец штриха', endpoint_contact:'Контакт концов'};
const statuses = {proposed:'Предложен', unknown:'Не определён', accepted:'Подтверждён'};
const $ = id => document.getElementById(id);
function el(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
function svg(tag, attrs={}) { const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v)); return n; }

export function createInspector({request,onReview}) {
  let current=null, serial=0, data=null, rows=[], selected=null, alternative=null, offset=0, imageUrl=null, controller=null;
  let comparison=null,dxfUrl=null,poll=null,viewBoxState=null,drag=null,moved=false,maskUrl=null,maskEpoch=0,polls=0;
  const lookup=new Map();
  function reset() {
    serial++;controller?.abort();controller=null;current=null;data=null;rows=[];selected=null;alternative=null;lookup.clear();
    clearTimeout(poll);poll=null;polls=0;comparison=null;viewBoxState=null;drag=null;
    if(dxfUrl)URL.revokeObjectURL(dxfUrl);dxfUrl=null;
    maskEpoch++;if(maskUrl)URL.revokeObjectURL(maskUrl);maskUrl=null;$('comparison-mask').value='none';
    for(const o of $('comparison-mask').options)if(o.value!=='none')o.disabled=true;
    $('comparison-state').textContent='';$('layer-dxf').disabled=true;
    if(imageUrl)URL.revokeObjectURL(imageUrl);imageUrl=null;
    $('inspection-canvas').replaceChildren();$('inspection-rows').replaceChildren();$('inspection-details').replaceChildren();
    $('inspection-summary').textContent='';$('inspection-source').textContent='';$('inspection-range').textContent='';$('inspection-download').disabled=true;
    $('inspection-previous').disabled=$('inspection-next').disabled=true;
    $('inspection-state').textContent='Выберите обработанный лист.';
  }
  async function context(ctx) {
    const key=`${ctx.key}:${ctx.page}`;
    if(current?.key===key)return;
    reset();if(!ctx.ready||!ctx.jobId)return;
    current={...ctx,key};const epoch=serial;controller=new AbortController();
    $('inspection-state').textContent='Подготавливаем наблюдения выбранного листа…';
    const base=`/v1/jobs/${encodeURIComponent(ctx.jobId)}/versions/${ctx.version}`;
    try {
      const received=await request(`${base}/pages/${ctx.page}/comparison`,{signal:controller.signal});
      if(epoch!==serial)return;
      comparison=received;
      const value=comparison.inspection;
      if(value.context.job_id!==ctx.jobId||value.context.version!==ctx.version||value.context.page_index!==ctx.page)throw new Error('Ответ относится к другой версии листа.');
      const blob=await request(`${base}/artifacts/${comparison.source.id}`,{signal:controller.signal},true);
      if(epoch!==serial)return;
      imageUrl=URL.createObjectURL(blob);data=value;
      for(const o of $('comparison-mask').options)if(o.value!=='none')o.disabled=!comparison.masks[o.value];
      for(const row of [...data.objects,...data.associations,...data.connectivity.endpoints,...data.connectivity.contacts])lookup.set(row.id,row);
      $('inspection-state').textContent=`Лист ${ctx.page+1} · версия ${ctx.version} · наблюдения требуют проверки`;
      $('inspection-summary').textContent=`Наблюдений ${data.counts.objects} · связей ${data.counts.associations}. Принято: примитивов ${comparison.verified_geometry_counts.primitives}, размеров ${comparison.verified_geometry_counts.dimensions}. ${data.connectivity.complete?'':'Поиск контактов ограничен. '}Конфликтов текста: ${data.source_conflicts.length}.${data.objects.length>1000?' На схеме первые 1000 объектов; полный список доступен справа.':''}`;
      $('inspection-source').textContent=JSON.stringify({physical_scale_confirmed:data.physical_scale_confirmed,
        source_coordinate_chain:data.source_coordinate_chain,source_completeness:data.source_completeness,
        source_conflicts:data.source_conflicts,source_notations:data.source_notations,context:data.context},null,2);
      $('inspection-download').disabled=false;offset=0;render();paint();
      await refreshDxf(base,ctx,epoch);
    } catch(error) { if(epoch===serial)$('inspection-state').textContent=error.status===409?'В этой версии нет растровых наблюдений или результат ещё не готов. Выберите режим анализа.':error.message; }
  }
  async function refreshDxf(base,ctx,epoch) {
    if(epoch!==serial||!comparison)return;
    const value=comparison;
    if(value.context.job_id!==ctx.jobId||value.context.version!==ctx.version||value.context.page_index!==ctx.page)throw new Error('Сравнение относится к другой версии.');
    const state=value.dxf.status;
    $('layer-dxf').disabled=state!=='ready';
    $('comparison-state').textContent=state==='ready'?
      `DXF: ${Object.entries(value.dxf.modelspace_counts).map(([kind,count])=>`${kind} ${count}`).join(', ')}. ${value.export_report.mode==='raster_ink_draft'?'Штриховая копия бинарной маски; OCR и CV показаны отдельно. ':''}${value.dxf.geometry_unrecovered?'Геометрия детали не восстановлена. ':''}Черновик, физический масштаб не подтверждён.`:
      state==='absent'?'DXF этой версией не создавался. Доступны исходник и наблюдения.':
      state==='counts_only'?`DXF: ${Object.entries(value.dxf.modelspace_counts).map(([kind,count])=>`${kind} ${count}`).join(', ')||'0 объектов'}. ${value.dxf.geometry_unrecovered?'Геометрия детали не восстановлена. ':''}Совмещение недоступно: система координат не подтверждена.`:
      ['pending','busy'].includes(state)?'Подготавливаем наложение фактического DXF…':'Совмещение DXF недоступно: '+(value.dxf.reason||'неподтверждённая система координат')+'. Обычный предпросмотр доступен во вкладке «Просмотр».';
    if(['pending','busy'].includes(state)) {
      if(++polls>60){$('comparison-state').textContent='Наложение пока недоступно. Повторите выбор версии для нового запроса.';return;}
      poll=setTimeout(async()=>{try {const next=await request(`${base}/pages/${ctx.page}/comparison`,{signal:controller.signal});
        if(epoch!==serial)return;if(next.context.manifest_sha256!==value.context.manifest_sha256)throw new Error('Манифест версии изменился.');
        comparison=next;await refreshDxf(base,ctx,epoch);}catch(error){if(epoch===serial)$('comparison-state').textContent=error.message;}},500);return;
    }
    if(state==='ready'&&!dxfUrl) {
      try {
        const blob=await request(`${base}/pages/${ctx.page}/comparison/dxf.svg`,{signal:controller.signal},true);
        if(epoch!==serial)return;dxfUrl=URL.createObjectURL(blob);paint();
      }catch(error){if(epoch===serial)$('comparison-state').textContent=error.message;}
    }
  }
  function filtered() {
    if(!data)return [];
    const kind=$('inspection-kind').value,status=$('inspection-status').value,query=$('inspection-search').value.toLocaleLowerCase();
    const source=kind==='endpoints'?data.connectivity.endpoints:kind==='contacts'?data.connectivity.contacts:[...data.objects,...data.associations];
    return source.filter(r=>(['all','endpoints','contacts'].includes(kind)||r.kind===kind)&&
      (status==='all'||r.status===status)&&`${r.id} ${r.source_id||''} ${r.raw||r.observation?.raw||''}`.toLocaleLowerCase().includes(query));
  }
  function render() {
    rows=filtered();offset=Math.max(0,Math.min(offset,Math.max(0,Math.floor((rows.length-1)/100)*100)));
    $('inspection-rows').replaceChildren();
    for(const row of rows.slice(offset,offset+100)) {
      const tr=el('tr');tr.classList.toggle('selected',row.id===selected?.id);
      const cell=el('td'),button=el('button',`${row.source_id||row.id}${row.observation?.raw||row.raw?' · '+(row.observation?.raw||row.raw):''}`,'inspection-object');
      button.type='button';button.title=row.id;button.addEventListener('click',()=>select(row));cell.append(button);
      tr.append(cell,el('td',labels[row.kind]||row.kind),el('td',statuses[row.status]||row.status));$('inspection-rows').append(tr);
    }
    if(!rows.length) { const td=el('td','Нет наблюдений по выбранному фильтру.');td.colSpan=3;const tr=el('tr');tr.append(td);$('inspection-rows').append(tr); }
    $('inspection-range').textContent=rows.length?`${offset+1}–${Math.min(offset+100,rows.length)} из ${rows.length}`:'0 объектов';
    $('inspection-previous').disabled=offset===0;$('inspection-next').disabled=offset+100>=rows.length;
  }
  function mapped(p) { const m=comparison?.frame.observation_to_canvas||data.preview.observation_to_preview;return [m[0][0]*p[0]+m[0][1]*p[1]+m[0][2],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]]; }
  function affine(m) {return `matrix(${m[0][0]} ${m[1][0]} ${m[0][1]} ${m[1][1]} ${m[0][2]} ${m[1][2]})`;}
  function paint(full=false) {
    const canvas=$('inspection-canvas');canvas.replaceChildren();if(!data||!imageUrl)return;
    const [w,h]=comparison?.frame.canvas_size_px||data.preview.size_px;let box=[-.5,-.5,w,h];
    if(selected&&!full) {
      const b=selected.bbox_px||[...selected.point_px,...selected.point_px];const corners=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]].map(mapped);
      const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]);const pad=Math.max(16,Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))*.18);
      const x=Math.max(0,Math.min(...xs)-pad),y=Math.max(0,Math.min(...ys)-pad),r=Math.min(w,Math.max(...xs)+pad),bottom=Math.min(h,Math.max(...ys)+pad);
      if(r>x&&bottom>y)box=[x,y,r-x,bottom-y];
    }
    if(full)viewBoxState=[-.5,-.5,w,h];if(viewBoxState)box=viewBoxState;else viewBoxState=box;
    canvas.setAttribute('viewBox',box.join(' '));
    if($('layer-source').checked) {const size=comparison?.source.size_px||[w,h];canvas.append(svg('image',{href:imageUrl,x:-.5,y:-.5,width:size[0],height:size[1],transform:affine(comparison?.frame.source_to_canvas||[[1,0,0],[0,1,0],[0,0,1]])}));}
    if(dxfUrl&&$('layer-dxf').checked)canvas.append(svg('image',{href:dxfUrl,x:0,y:0,width:comparison.dxf.svg_size[0],height:comparison.dxf.svg_size[1],transform:affine(comparison.dxf.svg_to_canvas),opacity:Number($('layer-opacity').value)/100,class:'comparison-dxf'}));
    const maskInfo=comparison?.masks[$('comparison-mask').value];
    if(maskUrl&&maskInfo) {
      const defs=svg('defs'),filter=svg('filter',{id:'comparison-mask-color','color-interpolation-filters':'sRGB'});
      filter.append(svg('feColorMatrix',{type:'matrix',values:`0 0 0 0 .95 0 0 0 0 .15 0 0 0 0 .05 ${maskInfo.ink_value===0?'-1 0 0 0 1':'1 0 0 0 0'}`}));defs.append(filter);canvas.append(defs);
      canvas.append(svg('image',{href:maskUrl,x:-.5,y:-.5,width:w,height:h,filter:'url(#comparison-mask-color)',opacity:.75}));
    }
    let group;
    const line=points=>group.append(svg('polyline',{points:points.map(mapped).map(p=>p.join(',')).join(' ')}));
    function mark(row) {
      if(row.kind==='geometry') {
        const g=row.observation;
        if(g.kind==='line')line(g.points_px);
        else { const sweep=g.kind==='circle'?360:g.sweep_angle_deg,start=g.start_angle_deg||0;line(Array.from({length:73},(_,i)=>{const a=(start+sweep*i/72)*Math.PI/180;return [g.center_px[0]+g.radius_px*Math.cos(a),g.center_px[1]+g.radius_px*Math.sin(a)];})); }
      } else if(row.kind==='text')line([...row.observation.polygon_px,row.observation.polygon_px[0]]);
      else if(row.kind==='arrow') {
        const a=row.observation,n=Math.hypot(...a.direction)||1;line([a.tip_px,[a.tip_px[0]-a.direction[0]*25/n,a.tip_px[1]-a.direction[1]*25/n]]);
        const p=mapped(a.tip_px);group.append(svg('circle',{cx:p[0],cy:p[1],r:Math.max(1,box[2]/100),fill:'#d34429'}));
      } else if(row.kind==='association') {
        const c=alternative===null?row.active:row.alternatives[alternative];if(c?.anchors_px?.length)line(c.anchors_px);
        const text=lookup.get(row.source_text_id);if(text)mark(text);
      } else if(row.point_px) { const p=mapped(row.point_px);group.append(svg('circle',{cx:p[0],cy:p[1],r:Math.max(1,box[2]/70)})); }
      else if(row.endpoint_ids)line(row.endpoint_ids.map(id=>lookup.get(id)?.point_px).filter(Boolean));
    }
    if($('layer-observations').checked)for(const row of data.objects.slice(0,1000)) {
      const color={geometry:'#d97816',text:'#bd308a',arrow:'#07803b'}[row.kind]||'#d97816';
      group=svg('g',{fill:'none',stroke:color,'stroke-width':Math.max(.7,box[2]/600),'data-object-id':row.id});
      group.style.cursor='pointer';group.addEventListener('click',()=>{if(!moved)select(row);});mark(row);canvas.append(group);
    }
    if(selected) {group=svg('g',{fill:'none',stroke:'#d34429','stroke-width':Math.max(1,box[2]/350)});mark(selected);canvas.append(group);}
    canvas.setAttribute('aria-label',`Совмещение ${selected?'объекта '+selected.id:'слоёв'}, лист ${current.page+1}, версия ${current.version}`);
  }
  function select(row) {
    selected=row;alternative=null;viewBoxState=null;render();const panel=$('inspection-details');panel.replaceChildren();
    panel.append(el('h4',row.id),el('p',`${labels[row.kind]||row.kind} · ${statuses[row.status]||row.status}. Принятие геометрии выполняется отдельно.`));
    if(row.reviewed_reading)panel.append(el('p',`Проверенное чтение: ${row.reviewed_reading.value}. Исходный OCR: ${row.observation.raw}. Краска скана сохранена.`));
    const home=el('button','Весь лист','text-button');home.type='button';home.addEventListener('click',()=>paint(true));
    const review=el('button','Указать ID в замечании','text-button');review.type='button';review.addEventListener('click',()=>onReview(row.id));panel.append(home,review);
    if(row.kind==='association') {
      panel.append(el('p',`Чтение: ${row.raw}. Тип: ${row.dimension_kind}. Единицы: ${row.nominal_unit_status==='degree_observation'?'градусы, наблюдение':'не подтверждены'}. Принадлежность виду не подтверждена.`));
      const choice=el('select');choice.setAttribute('aria-label','Гипотеза размерной связи');
      const active=el('option','Текущая гипотеза');active.value='';choice.append(active);
      row.alternatives.forEach((_,i)=>{const o=el('option',`Альтернатива ${i+1} из ${row.alternatives.length}`);o.value=i;choice.append(o);});
      const links=el('div',undefined,'inspection-links');panel.append(choice,links);
      function references() {
        links.replaceChildren();const c=alternative===null?row.active:row.alternatives[alternative];
        for(const id of [row.source_text_id,...Object.values(c?.references||{}).flat()]) {
          const button=el('button',id,'text-button');button.type='button';button.disabled=!lookup.has(id);
          button.addEventListener('click',()=>navigate(id));links.append(button);
        }
        if(c?.unresolved_references?.length)links.append(el('p','Часть ID не найдена в исходном контексте; гипотеза остаётся спорной.'));
      }
      choice.addEventListener('change',()=>{alternative=choice.value===''?null:Number(choice.value);references();paint();});references();
    }
    if(row.kind==='endpoint')panel.append(el('p',row.no_endpoint_neighbor?'Концевой сосед не найден в проверенной окрестности. Это не доказывает разрыв детали.':'Смысловое соединение не подтверждено.'));
    if(['endpoint','endpoint_contact'].includes(row.kind)) {
      const links=el('div',undefined,'inspection-links');
      for(const id of [row.object_id,...(row.object_ids||[]),...(row.endpoint_ids||[]),...(row.candidate_ids||[])].filter(Boolean)) {
        const button=el('button',id,'text-button');button.type='button';button.disabled=!lookup.has(id);
        button.addEventListener('click',()=>navigate(id));links.append(button);
      }
      panel.append(links);
    }
    const details=el('details'),summary=el('summary','Происхождение, причины и доказательства'),pre=el('pre',JSON.stringify(row,null,2));details.append(summary,pre);panel.append(details);
    paint();
  }
  function navigate(id) {
    const row=lookup.get(id);if(!row)return;
    $('inspection-kind').value=['endpoint','endpoint_contact'].includes(row.kind)?(row.kind==='endpoint'?'endpoints':'contacts'):row.kind;
    $('inspection-status').value='all';$('inspection-search').value=id;offset=0;select(row);
    $('inspection-canvas').scrollIntoView({block:'nearest'});
  }
  for(const name of ['kind','status','search'])$('inspection-'+name).addEventListener(name==='search'?'input':'change',()=>{offset=0;render();});
  for(const id of ['layer-source','layer-dxf','layer-observations','layer-opacity'])$(id).addEventListener(id==='layer-opacity'?'input':'change',()=>paint());
  $('comparison-mask').addEventListener('change',async()=>{
    const info=comparison?.masks[$('comparison-mask').value],epoch=serial,own=++maskEpoch;
    if(maskUrl)URL.revokeObjectURL(maskUrl);maskUrl=null;paint();if(!info||!current)return;
    try {const blob=await request(`/v1/jobs/${current.jobId}/versions/${current.version}/artifacts/${info.id}`,{signal:controller.signal},true);
      if(epoch!==serial||own!==maskEpoch)return;maskUrl=URL.createObjectURL(blob);paint();
    }catch(error){if(epoch===serial)$('comparison-state').textContent=error.message;}
  });
  function zoom(factor){if(!viewBoxState)return;const [x,y,w,h]=viewBoxState;const nw=Math.max(4,Math.min(100000,w*factor)),nh=h*nw/w;viewBoxState=[x+(w-nw)/2,y+(h-nh)/2,nw,nh];paint();}
  $('comparison-in').addEventListener('click',()=>zoom(.8));$('comparison-out').addEventListener('click',()=>zoom(1.25));$('comparison-fit').addEventListener('click',()=>paint(true));
  const canvas=$('inspection-canvas');canvas.addEventListener('wheel',e=>{if(!data)return;e.preventDefault();zoom(e.deltaY>0?1.15:1/1.15);},{passive:false});
  canvas.addEventListener('pointerdown',e=>{if(!viewBoxState)return;moved=false;drag={x:e.clientX,y:e.clientY,box:[...viewBoxState],inverse:canvas.getScreenCTM().inverse(),objectId:e.target.closest('[data-object-id]')?.dataset.objectId};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const a=new DOMPoint(drag.x,drag.y).matrixTransform(drag.inverse),b=new DOMPoint(e.clientX,e.clientY).matrixTransform(drag.inverse);if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>3)moved=true;viewBoxState=[drag.box[0]-(b.x-a.x),drag.box[1]-(b.y-a.y),drag.box[2],drag.box[3]];paint();});
  canvas.addEventListener('pointerup',()=>{const id=drag?.objectId;drag=null;if(!moved&&id&&lookup.has(id))select(lookup.get(id));});canvas.addEventListener('pointercancel',()=>{drag=null;});
  $('inspection-previous').addEventListener('click',()=>{offset-=100;render();});$('inspection-next').addEventListener('click',()=>{offset+=100;render();});
  $('inspection-download').addEventListener('click',()=>{if(!data)return;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=`observations-v${current.version}-page-${current.page+1}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);});
  reset();return {context,reset};
}
