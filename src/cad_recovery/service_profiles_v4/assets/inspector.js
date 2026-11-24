const NS = 'http://www.w3.org/2000/svg';
const labels = {geometry:'Геометрия', text:'Текст / знак', arrow:'Стрелка', association:'Размерная связь', endpoint:'Конец штриха', endpoint_contact:'Контакт концов'};
const statuses = {proposed:'Предложен', unknown:'Не определён', accepted:'Подтверждён'};
const $ = id => document.getElementById(id);
function el(tag, text, cls) { const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n; }
function svg(tag, attrs={}) { const n=document.createElementNS(NS,tag); for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v)); return n; }

export function createInspector({request,onReview}) {
  let current=null, serial=0, data=null, rows=[], selected=null, alternative=null, offset=0, imageUrl=null, controller=null;
  const lookup=new Map();
  function reset() {
    serial++;controller?.abort();controller=null;current=null;data=null;rows=[];selected=null;alternative=null;lookup.clear();
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
      const value=await request(`${base}/pages/${ctx.page}/inspection`,{signal:controller.signal});
      if(epoch!==serial)return;
      if(value.context.job_id!==ctx.jobId||value.context.version!==ctx.version||value.context.page_index!==ctx.page)throw new Error('Ответ относится к другой версии листа.');
      const blob=await request(`${base}/artifacts/${value.preview.artifact_id}`,{signal:controller.signal},true);
      if(epoch!==serial)return;
      imageUrl=URL.createObjectURL(blob);data=value;
      for(const row of [...data.objects,...data.associations,...data.connectivity.endpoints,...data.connectivity.contacts])lookup.set(row.id,row);
      $('inspection-state').textContent=`Лист ${ctx.page+1} · версия ${ctx.version} · наблюдения требуют проверки`;
      $('inspection-summary').textContent=`Объектов ${data.counts.objects} · связей ${data.counts.associations} · подтверждено ${data.counts.accepted}. Масштаб и принадлежность видам не подтверждены. ${data.connectivity.complete?'':'Поиск контактов ограничен. '}DrawingGraph 0.2: нужны проверенные параметры листа и координат.`;
      $('inspection-summary').textContent+=` Конфликтов текста: ${data.source_conflicts.length}. Полнота поиска и причины ограничений приведены ниже.`;
      $('inspection-source').textContent=JSON.stringify({physical_scale_confirmed:data.physical_scale_confirmed,
        source_coordinate_chain:data.source_coordinate_chain,source_completeness:data.source_completeness,
        source_conflicts:data.source_conflicts,source_notations:data.source_notations,context:data.context},null,2);
      $('inspection-download').disabled=false;offset=0;render();paint();
    } catch(error) { if(epoch===serial)$('inspection-state').textContent=error.status===409?'В этой версии нет растровых наблюдений или результат ещё не готов. Выберите режим анализа.':error.message; }
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
  function mapped(p) { const m=data.preview.observation_to_preview;return [m[0][0]*p[0]+m[0][1]*p[1]+m[0][2],m[1][0]*p[0]+m[1][1]*p[1]+m[1][2]]; }
  function paint(full=false) {
    const canvas=$('inspection-canvas');canvas.replaceChildren();if(!data||!imageUrl)return;
    const [w,h]=data.preview.size_px;let box=[0,0,w,h];
    if(selected&&!full) {
      const b=selected.bbox_px||[...selected.point_px,...selected.point_px];const corners=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]].map(mapped);
      const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]);const pad=Math.max(16,Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))*.18);
      const x=Math.max(0,Math.min(...xs)-pad),y=Math.max(0,Math.min(...ys)-pad),r=Math.min(w,Math.max(...xs)+pad),bottom=Math.min(h,Math.max(...ys)+pad);
      if(r>x&&bottom>y)box=[x,y,r-x,bottom-y];
    }
    canvas.setAttribute('viewBox',box.join(' '));canvas.append(svg('image',{href:imageUrl,x:0,y:0,width:w,height:h}));
    if(!selected)return;
    const group=svg('g',{fill:'none',stroke:'#d34429','stroke-width':Math.max(.7,box[2]/350),'vector-effect':'non-scaling-stroke'});
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
    mark(selected);canvas.append(group);canvas.setAttribute('aria-label',`Наблюдение ${selected.id}, лист ${current.page+1}, версия ${current.version}`);
  }
  function select(row) {
    selected=row;alternative=null;render();const panel=$('inspection-details');panel.replaceChildren();
    panel.append(el('h4',row.id),el('p',`${labels[row.kind]||row.kind} · ${statuses[row.status]||row.status}. Принятие геометрии выполняется отдельно.`));
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
  $('inspection-previous').addEventListener('click',()=>{offset-=100;render();});$('inspection-next').addEventListener('click',()=>{offset+=100;render();});
  $('inspection-download').addEventListener('click',()=>{if(!data)return;const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download=`observations-v${current.version}-page-${current.page+1}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);});
  reset();return {context,reset};
}
