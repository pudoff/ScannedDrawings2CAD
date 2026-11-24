import {operatorRequest} from './operator-key.js';
import {createTextReview} from './text-review.js';
const $=id=>document.getElementById(id);
const node=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e;};

export function createTables({request,onApplied}){
  request=operatorRequest(request,()=>$('table-key').value);
  let accessReady=false,keyEpoch=0,keyTimer=null,keyController=null;
  let current=null,packet=null,proposal=null,epoch=0,serial=0,timer=null,controller=null,start=null,image=null;
  let regions=[],urls=[];
  const canvas=$('table-canvas'),ctx=canvas.getContext('2d');
  const base=()=>`/v1/jobs/${current.jobId}/versions/${current.version}/pages/${current.page}/tables`;
  const textReview=createTextReview({request,base,onProposal:async row=>{serial++;$('table-reviewed').checked=false;await show(row,epoch);},draw,canvas,reviewer:()=>$('table-reviewer').value});
  function release(){for(const url of urls)URL.revokeObjectURL(url);urls=[];}
  function reset(){textReview.reset();epoch++;serial++;keyEpoch++;accessReady=false;clearTimeout(keyTimer);keyController?.abort();keyController=null;clearTimeout(timer);controller?.abort();controller=null;release();
    current=packet=proposal=image=null;start=null;regions=[];canvas.width=canvas.height=1;
    $('table-result').replaceChildren();$('table-regions').replaceChildren();$('table-state').textContent='Выберите обработанный документ.';
    $('table-prepare').disabled=$('table-apply').disabled=true;$('table-cancel').hidden=true;
    $('table-reviewed').checked=false;$('table-approval-wrap').hidden=true;$('table-key-wrap').hidden=true;
  }
  function changed(){textReview.reset();serial++;proposal=null;clearTimeout(timer);$('table-apply').disabled=true;$('table-reviewed').checked=false;
    $('table-approval-wrap').hidden=true;$('table-result').replaceChildren();controls();draw();}
  function controls(){
    const running=proposal&&['queued','running'].includes(proposal.status);
    for(const id of ['table-reviewer','table-page-kind','table-region-kind'])$(id).disabled=!!running;
    const allowed=accessReady&&packet?.permissions.can_apply;
    $('table-prepare').disabled=!allowed||!packet?.context.enabled||!regions.length||!$('table-reviewer').value.trim()||running;
    $('table-undo').disabled=!regions.length||running;$('table-cancel').hidden=!running;
    $('table-cancel').disabled=!allowed;
    $('table-apply').disabled=!(allowed&&proposal?.status==='prepared'&&$('table-reviewed').checked);
    $('table-key-wrap').hidden=!packet?.permissions.requires_accept_key;
  }
  function draw(preview){
    if(!image)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);
    for(const r of regions){ctx.strokeStyle=r.kind==='title'?'#b33ad8':'#006fea';ctx.lineWidth=Math.max(2,canvas.width/600);
      ctx.strokeRect(r.box_px[0],r.box_px[1],r.box_px[2]-r.box_px[0],r.box_px[3]-r.box_px[1]);}
    if(preview){ctx.strokeStyle='#ed8b17';ctx.strokeRect(preview[0],preview[1],preview[2]-preview[0],preview[3]-preview[1]);}
    $('table-regions').replaceChildren(...regions.map((r,i)=>node('div',`${i+1}. ${r.kind==='title'?'Основная надпись':'Таблица'} · ${r.box_px.join(', ')}`)));
  }
  function point(e){const b=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,Math.round((e.clientX-b.left)*canvas.width/b.width))),Math.max(0,Math.min(canvas.height,Math.round((e.clientY-b.top)*canvas.height/b.height)))];}
  function box(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
  canvas.onpointerdown=e=>{if(textReview.isDrawing()){start=point(e);canvas.setPointerCapture(e.pointerId);return;}if(!packet?.context.enabled||proposal&&['queued','running'].includes(proposal.status))return;start=point(e);canvas.setPointerCapture(e.pointerId);};
  canvas.onpointermove=e=>{if(start)draw(box(start,point(e)));};
  canvas.onpointercancel=()=>{start=null;draw();};
  canvas.onpointerup=e=>{if(!start)return;const b=box(start,point(e));start=null;if(textReview.isDrawing()){textReview.associate(b);draw(b);return;}
    if(b[2]-b[0]<2||b[3]-b[1]<2){draw();return;}
    if(regions.length>=16||regions.some(r=>b[0]<r.box_px[2]&&r.box_px[0]<b[2]&&b[1]<r.box_px[3]&&r.box_px[1]<b[3])){
      $('table-state').textContent='Области должны быть отдельными и не пересекаться. Максимум16.';draw();return;}
    regions.push({id:`region-${regions.length+1}`,kind:$('table-region-kind').value,box_px:b});changed();};
  $('table-undo').onclick=()=>{regions.pop();changed();};
  for(const id of ['table-reviewer','table-page-kind'])$(id).addEventListener('input',changed);
  $('table-reviewed').onchange=controls;
  $('table-key-wrap').firstChild.textContent='Ключ оператора';
  $('table-key').addEventListener('input',()=>{
    const own=epoch,credential=++keyEpoch;
    serial++;accessReady=false;clearTimeout(timer);clearTimeout(keyTimer);keyController?.abort();
    $('table-reviewed').checked=false;textReview.reset();controls();
    keyTimer=setTimeout(()=>refreshAccess(own,credential),250);
  });
  async function refreshAccess(own,credential){
    if(!current||!packet)return;
    const revision=serial;keyController=new AbortController();
    try{
      const value=await request(base(),{signal:keyController.signal});
      if(own!==epoch||credential!==keyEpoch)return;
      if(value.context.context_sha256!==packet.context.context_sha256)throw new Error('Документ изменился. Обновите лист.');
      packet={...packet,permissions:value.permissions};accessReady=true;
      $('table-state').textContent=value.permissions.can_apply?'Права оператора подтверждены.':
        'Для изменений введите действующий ключ оператора.';
      if(revision===serial){
        if(proposal){const latest=value.proposals.find(v=>v.id===proposal.id)||proposal;await show(latest,own,revision);}
        else if(value.published_text?.enabled)await textReview.showPublished(value.published_text);
      }
    }catch(error){if(own===epoch&&credential===keyEpoch&&error.name!=='AbortError')$('table-state').textContent=error.message;}
    finally{if(own===epoch&&credential===keyEpoch)controls();}
  }
  async function context(c){
    const key=`${c.key}:${c.page}`;if(current?.key===key)return;reset();if(!c.ready)return;
    current={...c,key};controller=new AbortController();const own=epoch,credential=keyEpoch;
    try{
      const value=await request(base(),{signal:controller.signal});if(own!==epoch)return;packet=value;accessReady=credential===keyEpoch;
      $('table-state').textContent=value.context.enabled?'Выделите таблицу или основную надпись. Виды снаружи сохранятся из текущего результата.':value.context.reason;
      if(value.context.working_size_px){
        const blob=await request(base()+'/image',{signal:controller.signal},true);if(own!==epoch)return;
        const url=URL.createObjectURL(blob);urls.push(url);const im=new Image();
        await new Promise((resolve,reject)=>{im.onload=resolve;im.onerror=reject;im.src=url;});if(own!==epoch)return;
        canvas.width=value.context.working_size_px[0];canvas.height=value.context.working_size_px[1];image=im;draw();
      }
      const latest=value.proposals[0];if(credential===keyEpoch&&latest){regions=latest.regions;$('table-reviewer').value=latest.reviewer;
        $('table-page-kind').value=latest.page_kind;draw();proposal=latest;await show(latest,own);}else if(credential===keyEpoch&&value.published_text?.enabled){await textReview.showPublished(value.published_text);}
      if(credential!==keyEpoch)refreshAccess(own,keyEpoch);
      controls();
    }catch(e){if(own===epoch)$('table-state').textContent=e.message;}
  }
  async function download(kind){
    try{const blob=await request(base()+`/${proposal.id}/files/${kind}`,{},true);const url=URL.createObjectURL(blob);
      const a=node('a');a.href=url;a.download=kind==='dxf'?`tables-page-${current.page+1}.dxf`:'tables.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    }catch(e){$('table-state').textContent=e.message;}
  }
  async function show(row,own,revision=serial){
    if(own!==epoch||revision!==serial)return;proposal=row;
    $('table-state').textContent=({queued:'Таблицы в очереди…',running:'Уточняем сетку и текст…',prepared:'Просмотр готов. Проверьте текст, сетку и сохранность видов.',
      rejected:'Пригодная сетка не найдена. Прежний DXF сохранён.',cancelled:'Обработка отменена; прежняя версия сохранена.',failed:'Обработка остановлена; прежняя версия сохранена.',applied:'Табличный результат сохранён отдельной версией.'})[row.status]||row.status;
    $('table-result').replaceChildren();$('table-approval-wrap').hidden=row.status!=='prepared';
    if(row.summary?.table_refinement){
      const detail=row.summary.table_refinement,list=node('ul');
      for(const r of detail.regions)list.append(node('li',`${r.id}: ${r.cells} ячеек; ${r.status==='unknown'?'не распознано — прежний CAD сохранён':`текст ${r.states.text_draft||0}, непрочитано ${r.states.unreadable||0}, пусто ${r.states.empty_observed||0}`}`));
      $('table-result').append(list,node('p','Чтения и знаки остаются черновыми. Неподходящую область удалите и подготовьте новый просмотр.'));
      if(detail.overlap_review_required)$('table-result').append(node('p','Спорные исходные примитивы сохранены; проверьте возможные наложения.'));
      if(['prepared','rejected','applied'].includes(row.status)){
        try{const blob=await request(base()+`/${row.id}/files/preview`,{},true);if(own!==epoch||revision!==serial)return;
          const url=URL.createObjectURL(blob);urls.push(url);const im=node('img');im.src=url;im.alt='Отрисовка предлагаемого DXF';$('table-result').append(im);
          const graph=await request(base()+`/${row.id}/files/graph`);if(own!==epoch||revision!==serial)return;
          const readings=node('div');readings.className='table-readings';
          for(const table of graph.tables)for(const cell of table.cells){if(cell.state==='empty_observed')continue;
            readings.append(node('div',`${table.id} · ${cell.id}: ${cell.tokens.map(t=>t.raw).join(' | ')||'не прочитано'}`));}
          $('table-result').append(readings);
        }catch(e){if(own===epoch)$('table-state').textContent=e.message;}
        const links=node('div');links.className='table-downloads';for(const kind of ['dxf','graph']){const b=node('button',kind==='dxf'?'Скачать DXF':'Отчёт JSON');b.type='button';b.onclick=()=>download(kind);links.append(b);}$('table-result').append(links);
      }
    }
    await textReview.show(row,current);
    if(row.status==='applied'){const b=node('button',`Открыть версию ${row.applied_version}`);b.id='table-open-version';b.onclick=()=>onApplied(row.applied_version);$('table-result').append(b);}
    controls();
    if(['queued','running'].includes(row.status))timer=setTimeout(()=>poll(own,revision),1000);
  }
  async function poll(own,revision){if(own!==epoch||revision!==serial||!proposal)return;try{const row=await request(base()+`/${proposal.id}`);if(own===epoch&&revision===serial)await show(row,own,revision);}catch(e){if(own===epoch&&revision===serial)$('table-state').textContent=e.message;}}
  $('table-prepare').onclick=async()=>{
    if(!packet)return;const own=epoch,edit=serial;
    try{const c=packet.context;const payload={context_sha256:c.context_sha256,source_manifest_sha256:c.source_manifest_sha256,
      source_sha256:c.source_sha256,binding_sha256:c.binding_sha256,page_kind:$('table-page-kind').value,reviewer:$('table-reviewer').value.trim(),regions};
      $('table-prepare').disabled=true;const row=await request(base()+'/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      if(own===epoch&&edit===serial)await show(row,own);
    }catch(e){if(own===epoch){$('table-state').textContent=e.message;controls();}}
  };
  $('table-cancel').onclick=async()=>{const own=epoch,revision=serial;try{const row=await request(base()+`/${proposal.id}/cancel`,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});if(own===epoch&&revision===serial)await show(row,own,revision);}catch(e){if(own===epoch&&revision===serial)$('table-state').textContent=e.message;}};
  $('table-apply').onclick=async()=>{if(!proposal||!$('table-reviewed').checked)return;const own=epoch;
    try{const row=await request(base()+`/${proposal.id}/apply`,{method:'POST',headers:{'Content-Type':'application/json',...($('table-key').value?{'X-Table-Accept-Key':$('table-key').value}:{})},
      body:JSON.stringify({preview_sha256:proposal.preview_sha256,reviewed:true})});if(own!==epoch)return;await onApplied(row.version);
    }catch(e){if(own===epoch)$('table-state').textContent=e.message;}
  };
  return {context,reset};
}
